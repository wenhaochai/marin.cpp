// Attention backends behind AttentionBackend (attention.hpp). DESIGN.md §3.4.
//   FA3      vendored FlashAttention-3, exec.attn_bwd_sched = persistent or doc_causal (fa3_backend.cu, sm_90a).
//   RefF32   portable reference (ref_attention.cu): f32 policy, tests, the oracle of the FA3 gates.
//   LevRef   not built (levanter's bf16 running statistics; the strict preset needs it, S4 gap in BUILD_LOG).
//   CudnnSdpa not built: E1b decides (DESIGN.md O1, BUILD_LOG decision 8).
#include <cuda_runtime.h>

#include <string>

#include "kernels/attention/attn_internal.hpp"
#include "marincpp/device.hpp"
#include "marincpp/kernels/attention.hpp"

namespace nm {

namespace {

void require_sm90(const char* what) {
  int dev = 0, major = 0;
  NM_CUDA(cudaGetDevice(&dev));
  NM_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  NM_REQUIRE(major == 9, std::string(what) + " needs an sm_90 GPU (built for sm_90a only); this device is sm_" +
                             std::to_string(major) + "x");
}

class Fa3Backend final : public AttentionBackend {
 public:
  AttnBackendKind kind() const override { return AttnBackendKind::FA3; }
  uint64_t workspace_bytes(const AttnShape& s, bool training) const override { return fa3::workspace_bytes(s, training); }
  void prepare(const AttnShape& s, bool training) override {
    require_sm90("FA3");
    (void)fa3::workspace_bytes(s, training);   // validates the shape
  }
  void fwd(const AttnShape& s, const AttnFwdArgs& a, void* ws, cudaStream_t st) override { fa3::fwd(s, a, ws, st); }
  void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st) override { fa3::bwd(s, a, ws, st); }
};

class RefF32Backend final : public AttentionBackend {
 public:
  AttnBackendKind kind() const override { return AttnBackendKind::RefF32; }
  uint64_t workspace_bytes(const AttnShape& s, bool training) const override { return ref::workspace_bytes(s, training); }
  void prepare(const AttnShape&, bool) override {}
  void fwd(const AttnShape& s, const AttnFwdArgs& a, void*, cudaStream_t st) override { ref::fwd(s, a, st); }
  void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st) override { ref::bwd(s, a, ws, st); }
};

}  // namespace

std::unique_ptr<AttentionBackend> make_attention(AttnBackendKind k, const DeviceContext&) {
  switch (k) {
    case AttnBackendKind::FA3: return std::make_unique<Fa3Backend>();
    case AttnBackendKind::RefF32: return std::make_unique<RefF32Backend>();
    case AttnBackendKind::LevRef:
      fatal(NM_WHERE, "attention backend levref is not built yet (levanter_strict needs it; docs/BUILD_LOG.md S4)");
    case AttnBackendKind::CudnnSdpa:
      fatal(NM_WHERE, "attention backend cudnn is not built: E1b did not require it (DESIGN.md O1)");
  }
  fatal(NM_WHERE, "unknown attention backend");
}

namespace fa3 {
void fwd(const AttnShape& s, const AttnFwdArgs& a, void* ws, cudaStream_t st) {
  require_sm90("FA3");
  fa3k::fwd(s, a, fa3_mode_of(s.bwd_sched), nullptr, ws, st);
}
void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st) {
  require_sm90("FA3");
  fa3k::bwd(s, a, fa3_mode_of(s.bwd_sched), nullptr, ws, st);
}
uint64_t workspace_bytes(const AttnShape& s, bool training) {
  return fa3k::workspace_bytes(s, fa3_mode_of(s.bwd_sched), training);
}
}  // namespace fa3

}  // namespace nm
