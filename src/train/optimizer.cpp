#include "train/optimizer.hpp"

#include <cuda_runtime.h>

#include <algorithm>

#include "marincpp/optim.hpp"

namespace nm::train {

Optimizer::Optimizer(const Config& c, const model::Baseline& net) {
  groups_ = opt::baseline_groups(c);
  const Ownership own = opt::plan_ownership_specs(net.params(), groups_, 1, false);
  tensors_ = opt::engine_layout(net.params(), groups_, own, 0);
  NM_REQUIRE(tensors_.size() == net.params().size(), "engine layout must have one tensor per parameter");
  tables_ = build_schedules(c);
  n_steps_ = static_cast<int64_t>(tables_.lr.size());
  NM_REQUIRE(n_steps_ > 0 && tables_.adam_lr.size() == tables_.lr.size(), "schedule tables are empty");
  total_ = net.total();
  // A first engine only to size its workspace: it requires non-null pointers, which it stores and never reads before
  // bind() rebuilds it with the arena's buffers.
  NM_CUDA(cudaMalloc(&dummy_, 256));
  std::vector<opt::EngTensor> ts = tensors_;
  for (auto& t : ts) t.p32 = t.s0 = t.s1 = static_cast<float*>(dummy_), t.g = static_cast<const float*>(dummy_);
  opt::EngOptions o;
  o.ns = optk::NsPrec::TF32;   // T1.0 item 5 (O3)
  o.sharded = false;
  engine_ = std::make_unique<opt::Engine>(groups_, ts, std::vector<RowPartition>{}, std::vector<const uint8_t*>{}, o);
}

Optimizer::~Optimizer() {
  engine_.reset();
  if (dummy_) cudaFree(dummy_);
}

void Optimizer::declare(MemoryPlan& plan) const {
  auto add = [&](const std::string& name, DType t, int64_t n) {
    BufferSpec b;
    b.name = name;
    b.group = name;
    b.life = Lifetime::Persistent;
    b.dtype = t;
    b.ndim = 1;
    b.shape[0] = n;
    plan.add(std::move(b));
  };
  add("opt.s0", DType::f32, total_);
  add("opt.s1", DType::f32, total_);
  for (const char* t : {"sched.lr", "sched.adam_lr", "sched.bc1", "sched.bc2"}) add(t, DType::f32, n_steps_);
  add("opt.step", DType::i32, 1);
  add("opt.ws", DType::u8, static_cast<int64_t>(std::max<size_t>(1, engine_->workspace_bytes())));
}

void Optimizer::bind(const Arena& a, const model::Baseline& net, cudaStream_t s, int32_t k0) {
  s0_ = a.bind_as<float>("opt.s0", DType::f32, {total_});
  s1_ = a.bind_as<float>("opt.s1", DType::f32, {total_});
  lr_ = a.bind_as<float>("sched.lr", DType::f32, {n_steps_});
  alr_ = a.bind_as<float>("sched.adam_lr", DType::f32, {n_steps_});
  bc1_ = a.bind_as<float>("sched.bc1", DType::f32, {n_steps_});
  bc2_ = a.bind_as<float>("sched.bc2", DType::f32, {n_steps_});
  step_ = a.bind_as<int32_t>("opt.step", DType::i32, {1});
  // the engine copied its tensor list at construction: rebuild it with the pointers
  for (size_t i = 0; i < tensors_.size(); ++i) {
    const int64_t o = net.offset(static_cast<int>(i));
    tensors_[i].p32 = net.p32() + o;
    tensors_[i].p16 = net.p16() + o;
    tensors_[i].g = net.grad() + o;
    tensors_[i].s0 = s0_ + o;
    tensors_[i].s1 = s1_ + o;
  }
  opt::EngOptions o;
  o.ns = optk::NsPrec::TF32;
  o.sharded = false;
  engine_ = std::make_unique<opt::Engine>(groups_, tensors_, std::vector<RowPartition>{},
                                          std::vector<const uint8_t*>{}, o);
  const TensorView ws = a.view("opt.ws");
  NM_REQUIRE(ws.nbytes() >= engine_->workspace_bytes(), "opt.ws is smaller than the engine workspace");
  engine_->bind(ws.data);
  NM_CUDA(cudaMemcpyAsync(lr_, tables_.lr.data(), n_steps_ * 4, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemcpyAsync(alr_, tables_.adam_lr.data(), n_steps_ * 4, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemcpyAsync(bc1_, tables_.bc1.data(), n_steps_ * 4, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemcpyAsync(bc2_, tables_.bc2.data(), n_steps_ * 4, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemcpyAsync(step_, &k0, 4, cudaMemcpyHostToDevice, s));
  NM_CUDA(cudaMemsetAsync(s0_, 0, size_t(total_) * 4, s));
  NM_CUDA(cudaMemsetAsync(s1_, 0, size_t(total_) * 4, s));
  engine_->set_tables({lr_, alr_}, bc1_, bc2_, step_, n_steps_);
  engine_->prepare(s);
  NM_CUDA(cudaStreamSynchronize(s));
}

void Optimizer::record_step(cudaStream_t s) { engine_->record_step(s, nullptr, true); }

}  // namespace nm::train
