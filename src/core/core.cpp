// Errors, dtype names, tensor views (core.hpp).
#include "marincpp/core.hpp"

#include <cuda_runtime.h>

#include <string>

namespace nm {

const char* dtype_name(DType t) {
  switch (t) {
    case DType::f32: return "f32";
    case DType::bf16: return "bf16";
    case DType::f16: return "f16";
    case DType::i32: return "i32";
    case DType::u32: return "u32";
    case DType::i64: return "i64";
    case DType::u8: return "u8";
    case DType::f64: return "f64";
  }
  return "?";
}

TensorView make_view(void* data, DType t, std::initializer_list<int64_t> shape) {
  NM_REQUIRE(shape.size() >= 1 && shape.size() <= 4, "a view has 1 to 4 dimensions");
  TensorView v;
  v.data = data;
  v.dtype = t;
  v.ndim = static_cast<int>(shape.size());
  int i = 0;
  for (int64_t s : shape) v.shape[i++] = s;
  int64_t st = 1;
  for (int d = v.ndim - 1; d >= 0; --d) {
    v.stride[d] = st;
    st *= v.shape[d];
  }
  return v;
}

[[noreturn]] void fatal(const char* where, const std::string& msg) { throw Fatal(std::string(where) + ": " + msg); }

namespace {
[[noreturn]] void device_error(const char* where, const std::string& msg) {
  throw DeviceError(std::string(where) + ": " + msg);
}
}  // namespace

void cuda_check(int err, const char* expr, const char* where) {
  if (err == 0) return;
  device_error(where, std::string(expr) + " -> " + cudaGetErrorName(static_cast<cudaError_t>(err)) + " (" +
                          cudaGetErrorString(static_cast<cudaError_t>(err)) + ")");
}

// NCCL and cuBLAS(Lt) are not linked by core; their status names are spelled out (nccl.h ncclResult_t, cublas_api.h
// cublasStatus_t).
void nccl_check(int err, const char* expr, const char* where) {
  if (err == 0) return;
  static const char* names[] = {"ncclSuccess",      "ncclUnhandledCudaError", "ncclSystemError",
                                "ncclInternalError", "ncclInvalidArgument",   "ncclInvalidUsage",
                                "ncclRemoteError",   "ncclInProgress"};
  const char* n = (err > 0 && err < 8) ? names[err] : "ncclResult?";
  device_error(where, std::string(expr) + " -> " + n + " (" + std::to_string(err) + ")");
}

void cublas_check(int err, const char* expr, const char* where) {
  if (err == 0) return;
  const char* n = "cublasStatus?";
  switch (err) {
    case 1: n = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
    case 3: n = "CUBLAS_STATUS_ALLOC_FAILED"; break;
    case 7: n = "CUBLAS_STATUS_INVALID_VALUE"; break;
    case 8: n = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
    case 11: n = "CUBLAS_STATUS_MAPPING_ERROR"; break;
    case 13: n = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
    case 14: n = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
    case 15: n = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
    case 16: n = "CUBLAS_STATUS_LICENSE_ERROR"; break;
  }
  device_error(where, std::string(expr) + " -> " + n + " (" + std::to_string(err) + ")");
}

}  // namespace nm
