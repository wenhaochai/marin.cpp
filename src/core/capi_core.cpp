// C ABI basics (capi.h): version, last error, device architecture.
#include <cuda_runtime.h>

#include <string>

#include "core/capi_util.hpp"
#include "marincpp/capi.h"
#include "nm_git_sha.h"   // generated at build time (cmake/GitSha.cmake): NM_GIT_SHA

namespace nm {
namespace {
thread_local std::string g_last_error;
}
void capi_set_error(const std::string& msg) { g_last_error = msg; }
void capi_clear_error() { g_last_error.clear(); }
}  // namespace nm

extern "C" {

const char* nm_version(void) { return NM_GIT_SHA; }

const char* nm_last_error(void) { return nm::g_last_error.c_str(); }

int nm_device_arch(void) {
  int dev = 0, major = 0, minor = 0;
  nm::capi_clear_error();
  cudaError_t e = cudaGetDevice(&dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  if (e != cudaSuccess) {
    nm::capi_set_error(std::string("nm_device_arch: ") + cudaGetErrorName(e) + " (" + cudaGetErrorString(e) + ")");
    cudaGetLastError();   // clear the error state for the caller's next CUDA call
    return 0;
  }
  return major * 10 + minor;
}

}  // extern "C"
