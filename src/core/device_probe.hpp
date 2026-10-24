// Build-structure probes (test s0_device_archs): one kernel from a portable unit and one from an sm_90a-only unit.
// Each returns the __CUDA_ARCH__ its kernel was compiled for (e.g. 800, 900), or 0 if the device has no image of it
// (cudaErrorNoKernelImageForDevice), which is the expected result for the sm_90a-only kernel on an A100.
#pragma once

namespace nm {
int probe_portable_kernel_arch();
int probe_sm90a_kernel_arch();
}  // namespace nm
