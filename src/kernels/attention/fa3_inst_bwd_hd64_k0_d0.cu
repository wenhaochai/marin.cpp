// FA3 backward variant bwd_hd64_k0_d0 (fa3_inst.cuh): kind 0 (stock varlen), deterministic false.
#include "kernels/attention/fa3_inst.cuh"

namespace nm::fa3k {
void bwd_hd64_k0_d0(Flash_bwd_params& p, bool gqa, cudaStream_t s) {
  if (gqa) {
    bwd_variant<64, 0, false, true>(p, s);
  } else {
    bwd_variant<64, 0, false, false>(p, s);
  }
}
}  // namespace nm::fa3k
