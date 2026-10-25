// FA3 backward variant bwd_hd128_k1_d0 (fa3_inst.cuh): kind 1 (work list), deterministic false.
#include <stdexcept>

#include "kernels/attention/fa3_inst.cuh"

namespace nm::fa3k {
void bwd_hd128_k1_d0(Flash_bwd_params& p, bool gqa, cudaStream_t s) {
  if (!gqa) { throw std::runtime_error("FA3 hd 128 is built for GQA only"); }
  bwd_variant<128, 1, false, true>(p, s);
}
}  // namespace nm::fa3k
