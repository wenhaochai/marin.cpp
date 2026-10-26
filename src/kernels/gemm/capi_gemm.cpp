// C ABI of the GEMM layer (capi.h "GEMM" block and capi_gemm.h). Test path: one GemmCache per device whose shapes
// are searched on first use with scratch from cudaMalloc; the trainer plans its shapes and calls choose_all once.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "core/capi_util.hpp"
#include "kernels/gemm/capi_gemm.h"
#include "kernels/gemm/gemm_internal.hpp"
#include "marincpp/capi.h"

namespace nm::gemm {

namespace {

struct DevCache {
  cublasLtHandle_t lt = nullptr;
  std::unique_ptr<GemmCache> cache;
  void* ws = nullptr;
  size_t ws_bytes = 0;
};

std::mutex g_mu;

DevCache& dev_cache() {
  static std::map<int, DevCache> m;
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  DevCache& d = m[dev];
  if (!d.cache) {
    NM_CUBLAS(cublasLtCreate(&d.lt));
    d.cache = std::make_unique<GemmCache>(d.lt, "");
  }
  return d;
}

size_t scratch_need(const GemmDesc& d) { return gemm_search_scratch_bytes(d); }

// Search scratch of the current device: kept and grown on demand (tests call the search thousands of times).
void* search_scratch(size_t need, cudaStream_t s) {
  struct Buf {
    void* p = nullptr;
    size_t n = 0;
  };
  static std::map<int, Buf> m;
  int dev = 0;
  NM_CUDA(cudaGetDevice(&dev));
  Buf& b = m[dev];
  if (b.n < need) {
    NM_CUDA(cudaStreamSynchronize(s));
    if (b.p) NM_CUDA(cudaFree(b.p));
    b.p = nullptr;
    b.n = 0;
    NM_CUDA(cudaMalloc(&b.p, need));
    b.n = need;
  }
  return b.p;
}

// Plans d, searches it if new (the device's search scratch), and makes the bound workspace large enough. Returns the
// id. Callers hold g_mu.
int ensure(DevCache& c, const GemmDesc& d, cudaStream_t s) {
  const int id = c.cache->plan(d);
  if (chosen_name(*c.cache, id).empty()) {
    const size_t need = scratch_need(d);
    void* scratch = search_scratch(need, s);
    NM_CUDA(cudaStreamSynchronize(s));
    c.cache->choose_all(s, scratch, need);
  }
  if (c.cache->workspace_bytes() > c.ws_bytes) {
    NM_CUDA(cudaStreamSynchronize(s));
    if (c.ws) NM_CUDA(cudaFree(c.ws));
    c.ws = nullptr;
    NM_CUDA(cudaMalloc(&c.ws, c.cache->workspace_bytes()));
    c.ws_bytes = c.cache->workspace_bytes();
  }
  c.cache->bind_workspace(c.ws);
  return id;
}

GemmDesc linear_desc(int orient, int rounding, int64_t T, int64_t N, int64_t K, int epilogue) {
  NM_REQUIRE(T > 0 && N > 0 && K > 0, "linear GEMM: T, N, K must be positive");
  switch (orient) {
    case 0: return linear_fwd_desc(T, N, K, static_cast<Epilogue>(epilogue), "capi.fwd");
    case 1: return linear_dgrad_desc(T, N, K, static_cast<Epilogue>(epilogue), "capi.dgrad");
    case 2:
      NM_REQUIRE(rounding == 0 || rounding == 1, "rounding must be 0 (levanter) or 1 (accurate)");
      return linear_wgrad_desc(T, N, K, rounding == 0 ? Rounding::Levanter : Rounding::Accurate, "capi.wgrad");
  }
  fatal(NM_WHERE, "orient must be 0 fwd, 1 dgrad, 2 wgrad");
}

// One timed search of d in a fresh cache (the device's search scratch): the shape's report.
ShapeReport search(const GemmDesc& d, cudaStream_t s) {
  std::lock_guard<std::mutex> g(g_mu);
  GemmCache cache(dev_cache().lt, "");
  cache.plan(d);
  const size_t need = scratch_need(d);
  std::vector<ShapeReport> reps;
  try {
    void* scratch = search_scratch(need, s);
    NM_CUDA(cudaStreamSynchronize(s));
    cache.choose_all(s, scratch, need);
    reps = report(cache);
  } catch (...) {
    forget(cache);
    throw;
  }
  forget(cache);
  NM_REQUIRE(!reps.empty(), "GEMM search: no report");
  return reps.back();
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    o += ch;
  }
  return o;
}

}  // namespace

GemmCache& capi_cache() { return *dev_cache().cache; }

}  // namespace nm::gemm

using namespace nm;
using namespace nm::gemm;

extern "C" {

int nm_linear_fwd(int epilogue, const void* x, const void* w, void* y, const void* epilogue_args, int64_t T, int64_t N,
                  int64_t K, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(epilogue == 0 && epilogue_args == nullptr, "nm_linear_fwd: fused epilogues are built in S8");
    std::lock_guard<std::mutex> g(g_mu);
    DevCache& c = dev_cache();
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int id = ensure(c, linear_desc(0, 1, T, N, K, epilogue), s);
    c.cache->run(id, x, w, nullptr, y, s);
  });
}

int nm_linear_dgrad(int epilogue, const void* dy, const void* w, void* dx, const void* epilogue_args, int64_t T,
                    int64_t N, int64_t K, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(epilogue == 0 && epilogue_args == nullptr, "nm_linear_dgrad: fused epilogues are built in S8");
    std::lock_guard<std::mutex> g(g_mu);
    DevCache& c = dev_cache();
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int id = ensure(c, linear_desc(1, 1, T, N, K, epilogue), s);
    c.cache->run(id, dy, w, nullptr, dx, s);
  });
}

int nm_linear_wgrad(int rounding, const void* dy, const void* x, float* dw, int64_t T, int64_t N, int64_t K,
                    void* stream) {
  return capi_call([&] {
    std::lock_guard<std::mutex> g(g_mu);
    DevCache& c = dev_cache();
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int id = ensure(c, linear_desc(2, rounding, T, N, K, 0), s);
    c.cache->run(id, dy, x, dw, dw, s);
  });
}

int nm_gemm_f32(int compute, const float* a, const float* b, float* d, int64_t m, int64_t n, int64_t k, int symmetric,
                void* stream) {
  return capi_call([&] {
    NM_REQUIRE(compute == 1 || compute == 2 || compute == 3, "compute must be 1 tf32, 2 f32 pedantic, 3 3xtf32");
    NM_REQUIRE(compute != 3, "nm_gemm_f32: 3xTF32 is not built yet (CUTLASS kernel, S4 gap)");
    GemmDesc dsc;
    dsc.m = m; dsc.n = n; dsc.k = k;
    dsc.op_a = Op::N; dsc.op_b = Op::N;
    dsc.a = dsc.b = dsc.c = dsc.d = DType::f32;
    dsc.lda = k; dsc.ldb = n; dsc.ldc = n; dsc.ldd = n;
    dsc.compute = compute == 1 ? Compute::TF32 : Compute::F32_PEDANTIC;
    dsc.symmetric_output = false;   // upper-triangle-only tiles are not built; the full product is the same matrix
    (void)symmetric;
    dsc.tag = "capi.f32";
    std::lock_guard<std::mutex> g(g_mu);
    DevCache& c = dev_cache();
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int id = ensure(c, dsc, s);
    c.cache->run(id, a, b, nullptr, d, s);
  });
}

int nm_gemm_search(int orient, int rounding, int64_t T, int64_t N, int64_t K, char* json_out, int json_cap,
                   void* stream) {
  return capi_call([&] {
    NM_REQUIRE(json_out != nullptr && json_cap > 0, "nm_gemm_search: json_out");
    const ShapeReport r = search(linear_desc(orient, rounding, T, N, K, 0), static_cast<cudaStream_t>(stream));
    std::ostringstream o;
    o << "{\"key\": \"" << json_escape(r.key) << "\", \"chosen\": \"" << json_escape(r.chosen) << "\", \"ms\": " << r.ms
      << ", \"tflops\": " << r.tflops << ", \"n_candidates\": " << r.n_candidates << ", \"top\": [";
    for (size_t i = 0; i < r.top.size(); ++i) {
      o << (i ? ", " : "") << "{\"name\": \"" << json_escape(r.top[i].name) << "\", \"ms\": " << r.top[i].ms
        << ", \"workspace\": " << r.top[i].workspace << "}";
    }
    o << "]}";
    const std::string js = o.str();
    NM_REQUIRE(int(js.size()) < json_cap, "nm_gemm_search: json_cap too small");
    std::snprintf(json_out, size_t(json_cap), "%s", js.c_str());
  });
}

int nm_gemm_chosen(int orient, int rounding, int64_t T, int64_t N, int64_t K, char* name_out, int name_cap) {
  return capi_call([&] {
    std::lock_guard<std::mutex> g(g_mu);
    DevCache& c = dev_cache();
    const int id = c.cache->plan(linear_desc(orient, rounding, T, N, K, 0));
    const std::string n = chosen_name(*c.cache, id);
    NM_REQUIRE(name_out != nullptr && int(n.size()) < name_cap, "nm_gemm_chosen: name_cap too small");
    std::snprintf(name_out, size_t(name_cap), "%s", n.c_str());
  });
}

int nm_gemm_candidates(int orient, int rounding, int64_t T, int64_t N, int64_t K, char* names_out, int names_cap,
                       void* stream) {
  return capi_call([&] {
    const ShapeReport r = search(linear_desc(orient, rounding, T, N, K, 0), static_cast<cudaStream_t>(stream));
    std::string names;
    for (const auto& c : r.all) names += c.name + "\n";
    NM_REQUIRE(names_out != nullptr && int(names.size()) < names_cap, "nm_gemm_candidates: names_cap too small");
    std::snprintf(names_out, size_t(names_cap), "%s", names.c_str());
  });
}

int nm_linear_named(const char* kernel, int orient, int rounding, const void* a, const void* b, void* d, int64_t T,
                    int64_t N, int64_t K, void* stream) {
  return capi_call([&] {
    NM_REQUIRE(kernel != nullptr && kernel[0] != 0, "nm_linear_named: kernel name");
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    std::lock_guard<std::mutex> g(g_mu);
    DevCache c;
    c.lt = dev_cache().lt;
    c.cache = std::make_unique<GemmCache>(c.lt, "");
    force_choice(*c.cache, kernel);
    try {
      const int id = ensure(c, linear_desc(orient, rounding, T, N, K, 0), s);
      c.cache->run(id, a, b, orient == 2 ? d : nullptr, d, s);
      NM_CUDA(cudaStreamSynchronize(s));
    } catch (...) {
      cudaStreamSynchronize(s);
      if (c.ws) cudaFree(c.ws);
      forget(*c.cache);
      throw;
    }
    if (c.ws) NM_CUDA(cudaFree(c.ws));
    forget(*c.cache);
  });
}

int nm_gemm_custom_count(void) { return int(custom_gemms().size()); }

const char* nm_gemm_custom_name(int i) {
  const auto& c = custom_gemms();
  return i >= 0 && size_t(i) < c.size() ? c[size_t(i)].name : "";
}

int nm_linear_custom(int i, int orient, int rounding, const void* a, const void* b, void* d, int64_t T, int64_t N,
                     int64_t K, void* stream) {
  return capi_call([&] {
    const auto& c = custom_gemms();
    NM_REQUIRE(i >= 0 && size_t(i) < c.size(), "nm_linear_custom: no custom kernel " + std::to_string(i));
    const GemmDesc dsc = linear_desc(orient, rounding, T, N, K, 0);
    NM_REQUIRE(c[size_t(i)].accepts(dsc), std::string("nm_linear_custom: ") + c[size_t(i)].name +
                                              " does not accept this shape");
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const size_t ws_bytes = c[size_t(i)].workspace ? c[size_t(i)].workspace(dsc) : 0;
    void* ws = nullptr;
    if (ws_bytes) NM_CUDA(cudaMallocAsync(&ws, ws_bytes, s));
    try {
      c[size_t(i)].launch(dsc, a, b, orient == 2 ? d : nullptr, d, ws, s);
    } catch (...) {
      if (ws) cudaFreeAsync(ws, s);
      throw;
    }
    if (ws) NM_CUDA(cudaFreeAsync(ws, s));
  });
}

}  // extern "C"
