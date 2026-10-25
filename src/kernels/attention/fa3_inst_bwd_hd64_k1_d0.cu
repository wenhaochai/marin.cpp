// FA3 backward variant bwd_hd64_k1_d0 (fa3_inst.cuh): kind 1 (work list), deterministic false.
#include "kernels/attention/fa3_inst.cuh"

namespace nm::fa3k {
void bwd_hd64_k1_d0(Flash_bwd_params& p, bool gqa, cudaStream_t s) {
  if (gqa) {
    bwd_variant<64, 1, false, true>(p, s);
  } else {
    bwd_variant<64, 1, false, false>(p, s);
  }
}
}  // namespace nm::fa3k
