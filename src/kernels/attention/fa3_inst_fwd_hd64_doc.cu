// FA3 forward variant fwd_hd64_doc (fa3_inst.cuh).
#include "kernels/attention/fa3_inst.cuh"

namespace nm::fa3k {
void fwd_hd64_doc(Flash_fwd_params& p, cudaStream_t s) { fwd_variant<64, false, false>(p, s); }
}  // namespace nm::fa3k
