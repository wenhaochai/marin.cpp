// Explicit FA3 kernel variants (included by the fa3_inst_*.cu units, one or two kernels per unit so the build
// parallelizes). Template arguments copy flash_fwd_launch_template.h run_mha_fwd_ and flash_bwd_launch_template.h
// run_mha_bwd_hdim64/128 for bf16, sm90, no softcap, no split, no paged KV.
#pragma once

#include "flash_bwd_launch_template.h"
#include "flash_fwd_launch_template.h"
#include "kernels/attention/fa3_internal.hpp"

namespace nm::fa3k {

using bf16_t = cutlass::bfloat16_t;

template <int kHeadDim, bool Varlen, bool PackGQA>
inline void fwd_variant(Flash_fwd_params& p, cudaStream_t s) {
  // Varlen: causal (VarlenDynamicPersistentTileScheduler). Batch: Is_local with the doc-causal arrays.
  constexpr bool Is_causal = Varlen;
  constexpr bool Is_local = !Varlen;
  run_flash_fwd<90, kHeadDim, kHeadDim, 1, bf16_t, bf16_t, Is_causal, Is_local, /*Has_softcap=*/false, Varlen,
                /*PagedKVNonTMA=*/false, /*AppendKV=*/false, /*HasQv=*/false, PackGQA, /*Split=*/false,
                /*V_colmajor=*/false>(p, s);
}

// kind: 0 stock varlen causal, 1 work-list varlen causal, 2 doc (batch, Is_local)
template <int kHeadDim, int kind, bool Det, bool GQA>
inline void bwd_variant(Flash_bwd_params& p, cudaStream_t s) {
  constexpr bool Varlen = kind != 2;
  constexpr bool Is_causal = kind != 2;
  constexpr bool Is_local = kind == 2;
  constexpr bool NmList = kind == 1;
  if constexpr (kHeadDim == 64) {
    run_flash_bwd<90, 64, 128, 128, bf16_t, Is_causal, Is_local, false, Varlen, Det, GQA, 2, 2, true, false, false, 2,
                  1, 2, 2, false, NmList>(p, s);
  } else {
    static_assert(kHeadDim == 128);
    run_flash_bwd<90, 128, 64, 128, bf16_t, Is_causal, Is_local, false, Varlen, Det, GQA, 2, 2, true, false, false, 2,
                  1, 2, 1, false, NmList>(p, s);
  }
}

}  // namespace nm::fa3k
