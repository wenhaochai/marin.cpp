// s0_arena and s0_device: GPU tests (skip without a CUDA device). Arena allocation, bind by name with dtype/shape
// checks, canary detection, NaN poison; the portable vs sm_90a-only build split.
#include <cuda_runtime.h>

#include <bit>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/device_probe.hpp"
#include "marincpp/capi.h"
#include "marincpp/memory.hpp"
#include "nmtest.hpp"

using namespace nm;

namespace {

void require_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    cudaGetLastError();
    SKIP("no CUDA device");
  }
}

BufferSpec buf(const std::string& name, Lifetime life, DType t, std::initializer_list<int64_t> shape, int layer = -1,
               Recompute rc = Recompute::Never) {
  BufferSpec b;
  b.name = name;
  b.life = life;
  b.dtype = t;
  b.ndim = static_cast<int>(shape.size());
  int i = 0;
  for (int64_t s : shape) b.shape[i++] = s;
  b.layer = layer;
  b.recompute = rc;
  return b;
}

MemoryPlan small_plan(bool canaries) {
  MemoryPlan p;
  p.add(buf("param", Lifetime::Persistent, DType::f32, {4, 8}));
  p.add(buf("odd", Lifetime::Persistent, DType::u8, {37}));
  p.add(buf("mb.t", Lifetime::Microbatch, DType::i32, {16}));
  // saved under remat=all although tagged Elementwise (build_plan declares it only when it is saved)
  p.add(buf("act.L0.a", Lifetime::Activation, DType::bf16, {8, 4}, 0, Recompute::Elementwise));
  p.add(buf("act.L0.x_in", Lifetime::Activation, DType::bf16, {8, 4}, 0, Recompute::Never));
  p.add(buf("tr.x", Lifetime::Transient, DType::f32, {64}));
  p.add(buf("ws.y", Lifetime::Workspace, DType::bf16, {33}));
  p.finalize(1 << 24, canaries);
  return p;
}

template <class T>
std::vector<T> download(const Arena& a, const std::string& name, DType t, std::initializer_list<int64_t> shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  std::vector<T> h(static_cast<size_t>(n));
  cudaMemcpy(h.data(), a.bind(name, t, shape), h.size() * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

}  // namespace

NM_TEST(s0_arena, bind_by_name) {
  require_gpu();
  MemoryPlan p = small_plan(false);
  Arena a;
  a.allocate(p, 0, ArenaDebug::Off);
  float* w = a.bind_as<float>("param", DType::f32, {4, 8});
  CHECK(w != nullptr);
  CHECK(reinterpret_cast<uintptr_t>(w) % kArenaAlign == 0);
  CHECK_THROWS(a.bind("param", DType::bf16, {4, 8}), "declared f32, bound as bf16");
  CHECK_THROWS(a.bind("param", DType::f32, {8, 4}), "declared shape [4, 8], bound as [8, 4]");
  CHECK_THROWS(a.bind("param", DType::f32, {32}), "declared shape");
  CHECK_THROWS(a.bind("nope", DType::f32, {1}), "no buffer named nope");
  TensorView v = a.view("ws.y");
  CHECK_EQ(v.numel(), int64_t(33));
  CHECK(v.dtype == DType::bf16);
  CHECK(a.free_bytes_at_start() > 0);
  a.check_canaries(nullptr);                     // Off mode: a no-op
  CHECK_EQ(a.canary_failure(), std::string(""));
  a.release();
}

NM_TEST(s0_arena, ownership) {
  static_assert(!std::is_copy_constructible_v<Arena> && !std::is_copy_assignable_v<Arena>);
  static_assert(std::is_nothrow_move_constructible_v<Arena>);
  require_gpu();
  MemoryPlan p = small_plan(true);
  Arena a;
  a.allocate(p, 0, ArenaDebug::Canary);
  void* w = a.bind("param", DType::f32, {4, 8});
  Arena b = std::move(a);                      // the allocation moves; `a` is empty
  CHECK(b.bind("param", DType::f32, {4, 8}) == w);
  CHECK_THROWS(a.bind("param", DType::f32, {4, 8}), "bind before allocate");
  CHECK_EQ(a.free_bytes_at_start(), uint64_t(0));
  a.release();                                 // releasing an empty arena is a no-op
  b.release();
  b.release();                                 // twice is safe
  CHECK_EQ(b.free_bytes_at_start(), uint64_t(0));
  b.allocate(p, 0, ArenaDebug::Off);           // released arenas can allocate again (debug mode reset)
  b.check_canaries(nullptr);
  CHECK_EQ(b.canary_failure(), std::string(""));
  CHECK_EQ(cudaGetLastError(), cudaSuccess);
}                                              // destructor releases `b`

NM_TEST(s0_arena, canary_detects_overrun) {
  require_gpu();
  MemoryPlan p = small_plan(true);
  Arena a;
  CHECK_THROWS(([&] { MemoryPlan q = small_plan(false); Arena b; b.allocate(q, 0, ArenaDebug::Canary); })(),
               "need a plan finalized with canaries");
  a.allocate(p, 0, ArenaDebug::Canary);
  a.check_canaries(nullptr);
  cudaDeviceSynchronize();
  CHECK_EQ(a.canary_failure(), std::string(""));
  // one byte past the end of "odd" (37 bytes)
  uint8_t* odd = a.bind_as<uint8_t>("odd", DType::u8, {37});
  cudaMemset(odd, 0, 38);
  a.check_canaries(nullptr);
  cudaDeviceSynchronize();
  CHECK_EQ(a.canary_failure(), std::string("odd"));
  a.release();
}

NM_TEST(s0_arena, canary_detects_underrun) {
  require_gpu();
  MemoryPlan p = small_plan(true);
  Arena a;
  a.allocate(p, 0, ArenaDebug::Canary);
  // one byte before "tr.x": with a band only after each buffer this byte was alignment padding
  uint8_t* x = a.bind_as<uint8_t>("tr.x", DType::f32, {64});
  cudaMemset(x - 1, 0, 1);
  a.check_canaries(nullptr);
  cudaDeviceSynchronize();
  CHECK_EQ(a.canary_failure(), std::string("tr.x"));
  a.release();
}

NM_TEST(s0_arena, poison_fills_transients_with_nan) {
  require_gpu();
  MemoryPlan p = small_plan(true);
  Arena a;
  a.allocate(p, 0, ArenaDebug::Poison);
  // saved state: persistent, microbatch and activation buffers (every Recompute tag) hold zeros before the poison
  for (const char* n : {"param", "mb.t", "act.L0.a", "act.L0.x_in"}) {
    TensorView v = a.view(n);
    cudaMemset(v.data, 0, v.numel() * dtype_size(v.dtype));
  }
  a.poison_transients(nullptr);
  cudaDeviceSynchronize();
  for (uint32_t x : download<uint32_t>(a, "tr.x", DType::f32, {64})) CHECK_EQ(x, 0x7fc00000u);
  for (uint16_t x : download<uint16_t>(a, "ws.y", DType::bf16, {33})) CHECK_EQ(x, uint16_t(0x7fc0));
  for (uint32_t x : download<uint32_t>(a, "param", DType::f32, {4, 8})) CHECK_EQ(x, 0u);
  for (uint32_t x : download<uint32_t>(a, "mb.t", DType::i32, {16})) CHECK_EQ(x, 0u);
  for (uint16_t x : download<uint16_t>(a, "act.L0.a", DType::bf16, {8, 4})) CHECK_EQ(x, uint16_t(0));
  for (uint16_t x : download<uint16_t>(a, "act.L0.x_in", DType::bf16, {8, 4})) CHECK_EQ(x, uint16_t(0));
  a.check_canaries(nullptr);
  cudaDeviceSynchronize();
  CHECK_EQ(a.canary_failure(), std::string(""));        // poison stays inside the buffers
  a.release();
}

NM_TEST(s0_device, portable_and_sm90a_images) {
  require_gpu();
  int arch = nm_device_arch();
  CHECK(arch >= 80);
  int portable = probe_portable_kernel_arch();
  int sm90a = probe_sm90a_kernel_arch();
  std::printf("    device sm_%d: portable kernel ran as __CUDA_ARCH__ %d; sm_90a-only kernel: %d\n", arch, portable,
              sm90a);
  if (arch == 90) {
    CHECK_EQ(portable, 900);
    CHECK_EQ(sm90a, 900);
  } else {
    CHECK_EQ(portable, 800);   // sm_80 SASS runs on sm_80/86/89 devices
    CHECK_EQ(sm90a, 0);        // no image: the sm_90a-only unit must not have been built for this device
  }
  CHECK(std::string(nm_version()).size() > 0);
  CHECK_EQ(std::string(nm_last_error()), std::string(""));
}
