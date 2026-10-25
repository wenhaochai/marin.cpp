// FA3 forward variant fwd_hd128_varlen_packgqa (fa3_inst.cuh).
#include "kernels/attention/fa3_inst.cuh"

namespace nm::fa3k {
void fwd_hd128_varlen_packgqa(Flash_fwd_params& p, cudaStream_t s) { fwd_variant<128, true, true>(p, s); }
}  // namespace nm::fa3k
