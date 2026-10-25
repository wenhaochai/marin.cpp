// Internal interface of src/kernels/attention: FA3 modes (training schedulers and the E1b bench arms) and the portable
// reference attention. No CUDA or FA3 header needed.
#pragma once

#include <cstdint>

#include "marincpp/kernels/attention.hpp"

namespace nm {

// FA3 launch modes. Doc = exec.attn_bwd_sched=doc_causal (the O1 decision, E1b). ListGrid = persistent: the device
// work list with one CTA per list slot (a fixed grid of rows * max_segs + T / 128 slots per head; slots past the list
// exit at once). E1b arms: ListPersistent = the same list on a grid of SM count with static striding (1.2-1.6x slower
// than ListGrid in E1b: no dynamic load balance); StockPadded = FA3's own varlen backward over the rows * max_segs
// segment slots (version 1); StockExact = FA3 with the real segment count and maximum length from the host (E1's
// setup; not capturable).
enum class Fa3Mode : int { ListPersistent = 0, Doc = 1, StockPadded = 2, StockExact = 3, ListGrid = 4 };
struct Fa3Exact {
  int n_segs = 0;       // real segment count (cu_seqlens has n_segs + 1 meaningful entries)
  int max_seqlen = 0;   // real longest segment
};

namespace fa3k {
uint64_t workspace_bytes(const AttnShape& s, Fa3Mode mode, bool training);
void fwd(const AttnShape& s, const AttnFwdArgs& a, Fa3Mode mode, const Fa3Exact* exact, void* ws, cudaStream_t st);
void bwd(const AttnShape& s, const AttnBwdArgs& a, Fa3Mode mode, const Fa3Exact* exact, void* ws, cudaStream_t st);
}  // namespace fa3k

inline Fa3Mode fa3_mode_of(AttnBwdSched b) {
  NM_REQUIRE(b == AttnBwdSched::Persistent || b == AttnBwdSched::DocCausal,
             "FA3: exec.attn_bwd_sched must be resolved (persistent or doc_causal)");
  return b == AttnBwdSched::Persistent ? Fa3Mode::ListGrid : Fa3Mode::Doc;
}

// Portable reference attention (ref_attention.cu; sm_80 and sm_90a): causal and same-segment softmax attention with
// f32 math and f32 running statistics over 1024-key blocks in ascending order (the RefF32 backend), bf16 or f32 storage.
// Segments come from cu_seqlens (packed starts, n_segs on the device or all rows * max_segs slots). Deterministic: every
// output element is one thread's fixed-order sum; dK/dV sum the G query heads of a group in ascending order.
// Workspace: softmax_d f32 [H, T] (backward only).
namespace ref {
uint64_t workspace_bytes(const AttnShape& s, bool training);
void fwd(const AttnShape& s, const AttnFwdArgs& a, cudaStream_t st);
void bwd(const AttnShape& s, const AttnBwdArgs& a, void* ws, cudaStream_t st);
}  // namespace ref

}  // namespace nm
