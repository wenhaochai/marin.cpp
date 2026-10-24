// C ABI of the data module (capi.h "data"): host threefry, the levanter order, a plan's sequence of (step, row).
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/capi_util.hpp"
#include "data/internal.hpp"
#include "data/threefry_impl.hpp"
#include "marincpp/capi.h"
#include "marincpp/config.hpp"
#include "marincpp/data.hpp"

extern "C" {

int nm_threefry2x32(uint32_t k0, uint32_t k1, uint32_t* x0, uint32_t* x1, int64_t n) {
  return nm::capi_call([&] {
    NM_REQUIRE(n >= 0 && (n == 0 || (x0 && x1)), "nm_threefry2x32: null buffers");
    for (int64_t i = 0; i < n; ++i) nm::tf::threefry2x32_20(k0, k1, x0[i], x1[i]);
  });
}

int nm_levanter_order(uint32_t data_seed, uint64_t n_seqs, int mix_block, uint64_t j0, uint64_t count,
                      uint32_t* out_host) {
  return nm::capi_call([&] {
    NM_REQUIRE(count == 0 || out_host, "nm_levanter_order: null output");
    nm::LevanterOrder o(data_seed, n_seqs, mix_block);
    for (uint64_t i = 0; i < count; ++i) out_host[i] = static_cast<uint32_t>(o(j0 + i));
  });
}

// config_toml: a config file path (as nm_schedule_tables takes it; "" = the defaults) or TOML text (any string with a
// newline); data.plan /
// data.dataset resolve against NM_DATA / NANOMARIN_ROOT like the trainer does.
int nm_plan_seq_of(const char* config_toml, int64_t step, int row, uint64_t* seq_host) {
  return nm::capi_call([&] {
    NM_REQUIRE(config_toml && seq_host, "nm_plan_seq_of: null argument");
    const std::string arg(config_toml);
    nm::Config c;
    if (arg.empty()) {
      c = nm::load_config({}, {}, {});   // the defaults (130m), as nm_schedule_tables
    } else if (arg.find('\n') == std::string::npos) {
      c = nm::load_config({arg}, {}, {});
    } else {
      const char* tmp = std::getenv("TMPDIR");
      std::string path = std::string(tmp && *tmp ? tmp : "/tmp") + "/nm_plan_XXXXXX";
      const int fd = ::mkstemp(path.data());
      NM_REQUIRE(fd >= 0, "nm_plan_seq_of: mkstemp failed");
      const bool ok = ::write(fd, arg.data(), arg.size()) == static_cast<ssize_t>(arg.size());
      ::close(fd);
      try {
        NM_REQUIRE(ok, "nm_plan_seq_of: temp file write failed");
        c = nm::load_config({path}, {}, {});
      } catch (...) {
        ::unlink(path.c_str());
        throw;
      }
      ::unlink(path.c_str());
    }
    const nm::DataPlan p = nm::load_data_plan(c.data_plan, c);
    *seq_host = p.seq_of(step, row);
  });
}

}  // extern "C"
