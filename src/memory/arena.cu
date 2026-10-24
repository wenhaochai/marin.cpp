// Arena: one cudaMalloc per GPU sized by the plan, bind by declared name with dtype and shape checks, canary and poison
// debug modes (memory.hpp; DESIGN.md §2.1-§2.2). Portable (sm_80 and sm_90a).
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "marincpp/memory.hpp"

namespace nm {
namespace {

constexpr uint8_t kCanaryByte = 0xA5;
constexpr int kNoFailure = 0x7f7f7f7f;   // cudaMemset byte 0x7f

__global__ void k_fill_guards(uint8_t* base, const uint64_t* guards, int n) {
  const int g = blockIdx.x;
  if (g >= n) return;
  uint8_t* p = base + guards[g];
  for (uint64_t i = threadIdx.x; i < kCanaryBytes; i += blockDim.x) p[i] = kCanaryByte;
}

__global__ void k_check_guards(const uint8_t* base, const uint64_t* guards, int n, int* bad) {
  const int g = blockIdx.x;
  if (g >= n) return;
  const uint8_t* p = base + guards[g];
  for (uint64_t i = threadIdx.x; i < kCanaryBytes; i += blockDim.x)
    if (p[i] != kCanaryByte) {
      atomicMin(bad, g);
      return;
    }
}

__global__ void k_fill_u32(uint32_t* p, uint64_t n, uint32_t v) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
    p[i] = v;
}
__global__ void k_fill_u16(uint16_t* p, uint64_t n, uint16_t v) {
  for (uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; i < n; i += uint64_t(gridDim.x) * blockDim.x)
    p[i] = v;
}

std::string shape_str(int ndim, const int64_t* s) {
  std::string o = "[";
  for (int i = 0; i < ndim; ++i) o += (i ? ", " : "") + std::to_string(s[i]);
  return o + "]";
}

// Guards 2i (before) and 2i+1 (after) belong to the i-th device buffer in declaration order.
std::vector<int> guarded_buffers(const MemoryPlan& p) {
  std::vector<int> ids;
  for (int i = 0; i < p.size(); ++i)
    if (!p.spec(BufferId{i}).host_pinned) ids.push_back(i);
  return ids;
}

}  // namespace

bool poison_target(const BufferSpec& b) {
  return !b.host_pinned && (b.life == Lifetime::Transient || b.life == Lifetime::Workspace);
}

Arena::~Arena() { release(); }

Arena::Arena(Arena&& o) noexcept { *this = std::move(o); }

Arena& Arena::operator=(Arena&& o) noexcept {
  if (this != &o) {
    release();
    plan_ = std::exchange(o.plan_, nullptr);
    dev_ = std::exchange(o.dev_, nullptr);
    host_ = std::exchange(o.host_, nullptr);
    dbg_ = std::exchange(o.dbg_, ArenaDebug::Off);
    free_at_start_ = std::exchange(o.free_at_start_, 0);
    device_ = std::exchange(o.device_, -1);
    d_guards_ = std::exchange(o.d_guards_, nullptr);
    n_guards_ = std::exchange(o.n_guards_, 0);
    d_bad_ = std::exchange(o.d_bad_, nullptr);
    h_bad_ = std::exchange(o.h_bad_, nullptr);
  }
  return *this;
}

void Arena::allocate(const MemoryPlan& plan, int device, ArenaDebug dbg) {
  NM_REQUIRE(dev_ == nullptr, "arena already allocated");
  NM_REQUIRE(plan.finalized(), "arena needs a finalized plan");
  NM_REQUIRE(dbg == ArenaDebug::Off || plan.canaries(), "canary/poison modes need a plan finalized with canaries");
  plan_ = &plan;
  dbg_ = dbg;
  device_ = device;
  NM_CUDA(cudaSetDevice(device));
  size_t free_b = 0, total_b = 0;
  NM_CUDA(cudaMemGetInfo(&free_b, &total_b));
  free_at_start_ = free_b;
  NM_CUDA(cudaMalloc(reinterpret_cast<void**>(&dev_), plan.device_bytes()));
  if (plan.pinned_bytes()) NM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&host_), plan.pinned_bytes(), 0));
  if (plan.canaries()) {
    std::vector<uint64_t> guards;
    for (int i : guarded_buffers(plan)) {
      guards.push_back(plan.offset(BufferId{i}) - kCanaryBytes);
      guards.push_back(plan.offset(BufferId{i}) + plan.spec(BufferId{i}).bytes());
    }
    n_guards_ = static_cast<int>(guards.size());
    NM_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_guards_), guards.size() * sizeof(uint64_t)));
    NM_CUDA(cudaMemcpy(d_guards_, guards.data(), guards.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    NM_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_bad_), sizeof(int)));
    NM_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&h_bad_), sizeof(int), 0));
    *h_bad_ = kNoFailure;
    if (n_guards_) k_fill_guards<<<n_guards_, 256>>>(reinterpret_cast<uint8_t*>(dev_), d_guards_, n_guards_);
    NM_CUDA(cudaGetLastError());
    NM_CUDA(cudaDeviceSynchronize());
  }
}

void Arena::release() {
  if (dev_ || host_ || d_guards_ || d_bad_ || h_bad_) {
    if (device_ >= 0) cudaSetDevice(device_);
    if (dev_) cudaFree(dev_);
    if (host_) cudaFreeHost(host_);
    if (d_guards_) cudaFree(d_guards_);
    if (d_bad_) cudaFree(d_bad_);
    if (h_bad_) cudaFreeHost(h_bad_);
  }
  dev_ = host_ = nullptr;
  d_guards_ = nullptr;
  d_bad_ = h_bad_ = nullptr;
  n_guards_ = 0;
  plan_ = nullptr;
  dbg_ = ArenaDebug::Off;
  free_at_start_ = 0;
  device_ = -1;
}

void* Arena::bind(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const {
  NM_REQUIRE(plan_ && dev_, "bind before allocate");
  BufferId id = plan_->find(name);
  if (!id.valid()) fatal(NM_WHERE, "bind: no buffer named " + name);
  const BufferSpec& s = plan_->spec(id);
  if (s.dtype != dtype)
    fatal(NM_WHERE, "bind " + name + ": declared " + dtype_name(s.dtype) + ", bound as " + dtype_name(dtype));
  bool same = static_cast<int>(shape.size()) == s.ndim;
  int i = 0;
  for (int64_t d : shape) same = same && i < s.ndim && s.shape[i++] == d;
  if (!same) {
    std::vector<int64_t> got(shape);
    fatal(NM_WHERE, "bind " + name + ": declared shape " + shape_str(s.ndim, s.shape) + ", bound as " +
                        shape_str(static_cast<int>(got.size()), got.data()));
  }
  return (s.host_pinned ? host_ : dev_) + plan_->offset(id);
}

TensorView Arena::view(const std::string& name) const {
  NM_REQUIRE(plan_ && dev_, "view before allocate");
  BufferId id = plan_->find(name);
  if (!id.valid()) fatal(NM_WHERE, "view: no buffer named " + name);
  const BufferSpec& s = plan_->spec(id);
  TensorView v;
  v.data = (s.host_pinned ? host_ : dev_) + plan_->offset(id);
  v.dtype = s.dtype;
  v.ndim = s.ndim;
  int64_t st = 1;
  for (int d = s.ndim - 1; d >= 0; --d) {
    v.shape[d] = s.shape[d];
    v.stride[d] = st;
    st *= s.shape[d];
  }
  return v;
}

void Arena::check_canaries(cudaStream_t s) const {
  if (dbg_ == ArenaDebug::Off || !n_guards_) return;
  NM_CUDA(cudaMemsetAsync(d_bad_, 0x7f, sizeof(int), s));
  k_check_guards<<<n_guards_, 256, 0, s>>>(reinterpret_cast<const uint8_t*>(dev_), d_guards_, n_guards_, d_bad_);
  NM_CUDA(cudaGetLastError());
  NM_CUDA(cudaMemcpyAsync(h_bad_, d_bad_, sizeof(int), cudaMemcpyDeviceToHost, s));
}

std::string Arena::canary_failure() const {
  if (!h_bad_ || *h_bad_ == kNoFailure) return "";
  std::vector<int> ids = guarded_buffers(*plan_);
  const int g = *h_bad_ / 2;
  return (*h_bad_ >= 0 && g < static_cast<int>(ids.size())) ? plan_->spec(BufferId{ids[g]}).name : "?";
}

void Arena::poison_transients(cudaStream_t s) const {
  if (dbg_ != ArenaDebug::Poison) return;
  for (int i = 0; i < plan_->size(); ++i) {
    const BufferSpec& b = plan_->spec(BufferId{i});
    if (!poison_target(b)) continue;
    char* p = dev_ + plan_->offset(BufferId{i});
    const uint64_t n = static_cast<uint64_t>(b.numel());
    switch (b.dtype) {
      case DType::f32: k_fill_u32<<<256, 256, 0, s>>>(reinterpret_cast<uint32_t*>(p), n, 0x7fc00000u); break;
      case DType::bf16: k_fill_u16<<<256, 256, 0, s>>>(reinterpret_cast<uint16_t*>(p), n, 0x7fc0u); break;
      case DType::f16: k_fill_u16<<<256, 256, 0, s>>>(reinterpret_cast<uint16_t*>(p), n, 0x7e00u); break;
      default: NM_CUDA(cudaMemsetAsync(p, 0xff, b.bytes(), s)); break;
    }
    NM_CUDA(cudaGetLastError());
  }
}

}  // namespace nm
