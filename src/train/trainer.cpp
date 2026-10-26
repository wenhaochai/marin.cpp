#include "train/trainer.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <barrier>
#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "marincpp/data.hpp"
#include "marincpp/kernels/gemm.hpp"
#include "marincpp/memory.hpp"
#include "model/baseline.hpp"
#include "model/fixture.hpp"
#include "train/optimizer.hpp"

namespace nm::train {

int train(const TrainOptions& o) {
  Config c;
  apply_preset(c, o.size);
  const ModelShape m = model_shape(c);
  const int n = o.gpus, B = c.data_global_batch, S = m.S;
  NM_REQUIRE(n >= 1 && n <= kMaxGpus, "gpus must be 1..8");
  NM_REQUIRE(B % n == 0, "global batch must divide by the GPU count");
  const int spr = B / n;
  NM_REQUIRE(spr % o.micro_rows == 0, "micro_rows must divide the rows per rank");
  const int n_micro = spr / o.micro_rows;
  NM_REQUIRE(n_micro <= model::Baseline::kMaxMicro, "too many microbatches");
  NM_REQUIRE(o.steps >= 1 && o.steps <= c.optim_num_steps, "steps must be 1..optim.num_steps");

  const DatasetMeta meta = load_dataset_meta(o.data_dir);
  NM_REQUIRE(meta.seq_len == S, "dataset window length differs from the model's");
  TokenFile tf;
  tf.open(meta);
  const std::vector<float> p0 = model::load_fixture_storage(o.init_fixture, model::baseline_params(m));
  std::vector<bf16_t> p0_16(p0.size());
  for (size_t i = 0; i < p0.size(); ++i) p0_16[i].bits = f32_to_bf16_bits(p0[i]);

  std::vector<ncclComm_t> comms(n);
  std::vector<int> devs(n);
  for (int r = 0; r < n; ++r) devs[r] = r;
  NM_NCCL(ncclCommInitAll(comms.data(), n, devs.data()));

  std::barrier bar(n);
  std::mutex mu;
  std::exception_ptr err;
  std::FILE* jsonl = o.metrics.empty() ? nullptr : std::fopen(o.metrics.c_str(), "w");

  auto worker = [&](int r) {
    try {
      NM_CUDA(cudaSetDevice(r));
      cudaStream_t s;
      NM_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      cublasLtHandle_t lt;
      NM_CUBLAS(cublasLtCreate(&lt));
      const LevanterOrder order(static_cast<uint32_t>(c.data_data_seed), meta.n_seqs, c.data_mix_block);  // per rank:
                                                                                                    // mutable cache
      model::BaselineSpec bs;
      bs.m = m;
      bs.micro_rows = o.micro_rows;
      bs.global_rows = B;
      bs.max_segs = 24;
      bs.ce_chunk = o.ce_chunk;
      bs.num.rounding = o.rounding;
      bs.num.embed_accum = o.embed;
      bs.attn = o.attn;
      bs.ce = o.ce;
      model::Baseline net(bs);
      GemmCache gc(lt, o.gemm_cache);
      net.plan_gemms(gc);
      // rank 0 searches and saves; the others load its choices, so every rank runs the same kernels (bitwise ranks)
      auto choose = [&] {
        void* scratch = nullptr;
        NM_CUDA(cudaMalloc(&scratch, net.gemm_search_scratch()));
        gc.choose_all(s, scratch, net.gemm_search_scratch());
        NM_CUDA(cudaStreamSynchronize(s));
        NM_CUDA(cudaFree(scratch));
      };
      if (r == 0) {
        choose();
        gc.save();
      }
      bar.arrive_and_wait();
      if (r != 0) choose();

      Optimizer opt(c, net);
      MemoryPlan plan;
      net.declare(plan, gc.workspace_bytes());
      opt.declare(plan);
      {
        BufferSpec b;
        b.name = b.group = "trainer.tokens";
        b.life = Lifetime::Persistent;
        b.dtype = DType::i32;
        b.ndim = 2;
        b.shape[0] = spr, b.shape[1] = S;
        plan.add(std::move(b));
      }
      size_t free_b = 0, total_b = 0;
      NM_CUDA(cudaMemGetInfo(&free_b, &total_b));
      plan.finalize(free_b - (size_t(2) << 30), false);
      Arena arena;
      arena.allocate(plan, r, ArenaDebug::Off);
      net.bind(arena, gc, s);
      opt.bind(arena, net, s, 0);
      int32_t* dtok = arena.bind_as<int32_t>("trainer.tokens", DType::i32, {spr, S});
      NM_CUDA(cudaMemcpy(net.p32(), p0.data(), p0.size() * 4, cudaMemcpyHostToDevice));
      NM_CUDA(cudaMemcpy(net.p16(), p0_16.data(), p0_16.size() * 2, cudaMemcpyHostToDevice));
      if (r == 0)
        std::printf("arena %.2f GiB per GPU; %d GPUs x %d microbatches of %d rows; graph %s\n",
                    plan.device_bytes() / 1073741824.0, n, n_micro, o.micro_rows, o.graph ? "on" : "off");

      // tokens of step k for this rank: rows [r*spr, (r+1)*spr) of the global batch, double-buffered in pinned memory
      int32_t* pin[2];
      cudaEvent_t copied[2];
      for (int i = 0; i < 2; ++i) {
        NM_CUDA(cudaHostAlloc(&pin[i], size_t(spr) * S * 4, cudaHostAllocDefault));
        NM_CUDA(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming));
        NM_CUDA(cudaEventRecord(copied[i], s));
      }
      auto fill = [&](int64_t k, int32_t* dst) {
        for (int i = 0; i < spr; ++i) {
          const uint32_t* w = tf.window(order(uint64_t(k) * B + uint64_t(r) * spr + i), S);
          for (int j = 0; j < S; ++j) dst[size_t(i) * S + j] = static_cast<int32_t>(w[j]);
        }
      };
      float* loss_mb = const_cast<float*>(net.loss_slots());
      auto record_step = [&] {
        net.zero_grads(s);
        for (int j = 0; j < n_micro; ++j) net.record_microbatch(j, dtok + size_t(j) * o.micro_rows * S, s);
        NM_NCCL(ncclAllReduce(net.grad(), net.grad(), static_cast<size_t>(net.total()), ncclFloat, ncclSum, comms[r], s));
        NM_NCCL(ncclAllReduce(loss_mb, loss_mb, static_cast<size_t>(n_micro), ncclFloat, ncclSum, comms[r], s));
        opt.record_step(s);
      };

      // warm-up outside any capture: every kernel's first launch (lazy init) and the collectives, no update
      fill(0, pin[0]);
      NM_CUDA(cudaMemcpyAsync(dtok, pin[0], size_t(spr) * S * 4, cudaMemcpyHostToDevice, s));
      net.zero_grads(s);
      net.record_microbatch(0, dtok, s);
      NM_NCCL(ncclAllReduce(net.grad(), net.grad(), static_cast<size_t>(net.total()), ncclFloat, ncclSum, comms[r], s));
      NM_NCCL(ncclAllReduce(loss_mb, loss_mb, static_cast<size_t>(n_micro), ncclFloat, ncclSum, comms[r], s));
      NM_CUDA(cudaStreamSynchronize(s));
      net.zero_grads(s);
      NM_CUDA(cudaStreamSynchronize(s));

      cudaGraphExec_t ge = nullptr;
      if (o.graph) {
        cudaGraph_t g;
        NM_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        record_step();
        NM_CUDA(cudaStreamEndCapture(s, &g));
        NM_CUDA(cudaGraphInstantiate(&ge, g, 0));
        NM_CUDA(cudaGraphDestroy(g));
      }
      bar.arrive_and_wait();

      auto t_prev = std::chrono::steady_clock::now();
      int64_t k_prev = 0;
      std::vector<float> slots(n_micro);
      for (int64_t k = 0; k < o.steps; ++k) {
        int32_t* buf = pin[k % 2];
        if (k > 0) {
          NM_CUDA(cudaEventSynchronize(copied[k % 2]));   // the copy that last read this buffer is done
          fill(k, buf);
        }
        NM_CUDA(cudaMemcpyAsync(dtok, buf, size_t(spr) * S * 4, cudaMemcpyHostToDevice, s));
        NM_CUDA(cudaEventRecord(copied[k % 2], s));
        if (ge) NM_CUDA(cudaGraphLaunch(ge, s));
        else record_step();
        const bool log = (k + 1) % o.log_every == 0 || k == 0 || k + 1 == o.steps;
        if (log) {
          NM_CUDA(cudaStreamSynchronize(s));
          const auto t_now = std::chrono::steady_clock::now();
          const double dt = std::chrono::duration<double>(t_now - t_prev).count() / double(k + 1 - k_prev);
          t_prev = t_now, k_prev = k + 1;
          if (r == 0) {
            NM_CUDA(cudaMemcpy(slots.data(), loss_mb, n_micro * 4, cudaMemcpyDeviceToHost));
            double sum = 0;
            for (float x : slots) sum += x;
            const double loss = sum * net.loss_scale();
            const double tps = double(B) * S / dt;
            std::lock_guard<std::mutex> l(mu);
            std::printf("step %6lld  loss %.6f  %.4f s/step  %.0f tok/s  (%.0f tok/s/GPU)\n",
                        static_cast<long long>(k), loss, dt, tps, tps / n);
            std::fflush(stdout);
            if (jsonl) {
              std::fprintf(jsonl, "{\"_step\": %lld, \"train/loss\": %.9g, \"throughput/duration\": %.6g, "
                                  "\"throughput/tokens_per_second\": %.6g}\n",
                           static_cast<long long>(k), loss, dt, tps);
              std::fflush(jsonl);
            }
          }
        }
      }
      NM_CUDA(cudaStreamSynchronize(s));
      if (ge) cudaGraphExecDestroy(ge);
      for (int i = 0; i < 2; ++i) cudaFreeHost(pin[i]), cudaEventDestroy(copied[i]);
      cublasLtDestroy(lt);
      cudaStreamDestroy(s);
    } catch (...) {
      std::lock_guard<std::mutex> l(mu);
      if (!err) err = std::current_exception();
    }
  };
  std::vector<std::thread> ts;
  for (int r = 0; r < n; ++r) ts.emplace_back(worker, r);
  for (auto& t : ts) t.join();
  for (auto& cm : comms) ncclCommDestroy(cm);
  if (jsonl) std::fclose(jsonl);
  if (err) std::rethrow_exception(err);
  return 0;
}

}  // namespace nm::train
