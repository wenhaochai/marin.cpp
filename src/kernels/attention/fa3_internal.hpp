// FA3 glue internal to src/kernels/attention (sm_90a units only): the kernel variants marincpp instantiates from the
// vendored FlashAttention-3 (third_party/fa3-hopper, see its README for the patches) and the dispatch to them.
#pragma once

#include <cuda_runtime.h>

#include "flash.h"

namespace nm::fa3k {

// Forward: Varlen = packed segments (cu_seqlens, VarlenDynamicPersistentTileScheduler, causal); Doc = batch of rows with
// the doc-causal patch (Is_local + doc_lo / doc_hi, DynamicPersistentTileScheduler).
enum class FwdKind { Varlen = 0, Doc = 1 };
// Backward: Stock = FA3's own varlen causal backward (SingleTileBwdLPTScheduler; E1b "padded" and "exact" arms only);
// List = varlen causal over marincpp's device work list (exec.attn_bwd_sched=persistent); Doc = batch of rows with the
// doc-causal patch (exec.attn_bwd_sched=doc_causal).
enum class BwdKind { Stock = 0, List = 1, Doc = 2 };

// Tile shapes of the instantiations (flash_fwd_launch_template.h tile_size_fwd_sm90, flash_bwd_launch_template.h
// run_mha_bwd_hdim64/128 for causal / local).
constexpr int fwd_block_m(int hd) { return hd <= 64 ? 192 : 128; }
constexpr int bwd_block_m(int hd) { return hd <= 64 ? 128 : 64; }
constexpr int bwd_block_n(int) { return 128; }

// One function per instantiation unit (fa3_inst_*.cu); params as flash_api.cpp fills them. Backward units: hd 64 has
// MHA and GQA kernels, hd 128 GQA only (the owner shapes, DESIGN.md §3.4); kind k0 Stock, k1 List, k2 Doc; d0/d1 =
// deterministic off/on.
void fwd_hd64_varlen(Flash_fwd_params& p, cudaStream_t s);
void fwd_hd64_varlen_packgqa(Flash_fwd_params& p, cudaStream_t s);
void fwd_hd64_doc(Flash_fwd_params& p, cudaStream_t s);
void fwd_hd128_varlen_packgqa(Flash_fwd_params& p, cudaStream_t s);
void fwd_hd128_doc(Flash_fwd_params& p, cudaStream_t s);
#define NM_FA3_BWD_UNIT(hd, k, d) void bwd_hd##hd##_k##k##_d##d(Flash_bwd_params& p, bool gqa, cudaStream_t s);
NM_FA3_BWD_UNIT(64, 0, 0) NM_FA3_BWD_UNIT(64, 0, 1) NM_FA3_BWD_UNIT(64, 1, 0) NM_FA3_BWD_UNIT(64, 1, 1)
NM_FA3_BWD_UNIT(64, 2, 0) NM_FA3_BWD_UNIT(64, 2, 1)
NM_FA3_BWD_UNIT(128, 0, 0) NM_FA3_BWD_UNIT(128, 0, 1) NM_FA3_BWD_UNIT(128, 1, 0) NM_FA3_BWD_UNIT(128, 1, 1)
NM_FA3_BWD_UNIT(128, 2, 0) NM_FA3_BWD_UNIT(128, 2, 1)
#undef NM_FA3_BWD_UNIT

}  // namespace nm::fa3k
