// C ABI of the schedules (capi.h: nm_schedule_tables), for the T0.7 gate and other Python tests.
//   config_toml   path of a marincpp TOML config (model.size picks the size preset); NULL or "" = the defaults (130m)
//   n             table length; must equal optim.num_steps of that config
//   from_fixture  1: load_schedules (optim.schedule_fixture must be set; the fixture is checked against the config and
//                    the formula, *n_ulp_diffs = entries where the formula differs over all four tables, or -1 when
//                    the formula refuses the config: cross-check unavailable, the fixture tables are still returned)
//                 0: build_schedules, the formula alone (*n_ulp_diffs = 0)
//   lr_host, adam_lr_host [n]: eta(k), eta_a(k); bc1_host, bc2_host [n]: 1 - b^t at t = k + 1. Every output may be NULL.
#include <algorithm>
#include <string>
#include <vector>

#include "core/capi_util.hpp"
#include "marincpp/capi.h"
#include "marincpp/config.hpp"
#include "marincpp/optim.hpp"

extern "C" int nm_schedule_tables(const char* config_toml, int64_t n, int from_fixture, float* lr_host,
                                  float* adam_lr_host, float* bc1_host, float* bc2_host, int* n_ulp_diffs) {
  return nm::capi_call([&] {
    std::vector<std::string> files;
    if (config_toml && *config_toml) files.emplace_back(config_toml);
    const nm::Config c = nm::load_config(files, {}, {});
    NM_REQUIRE(n == c.optim_num_steps,
               "n = " + std::to_string(n) + " but optim.num_steps = " + std::to_string(c.optim_num_steps));
    NM_REQUIRE(from_fixture == 0 || from_fixture == 1, "from_fixture must be 0 or 1");
    NM_REQUIRE(from_fixture == 0 || !c.optim_schedule_fixture.empty(), "from_fixture = 1 needs optim.schedule_fixture");
    const nm::ScheduleTables t = from_fixture ? nm::load_schedules(c) : nm::build_schedules(c);
    auto out = [](const std::vector<float>& v, float* dst) {
      if (dst) std::copy(v.begin(), v.end(), dst);
    };
    out(t.lr, lr_host);
    out(t.adam_lr, adam_lr_host);
    out(t.bc1, bc1_host);
    out(t.bc2, bc2_host);
    if (n_ulp_diffs) *n_ulp_diffs = t.n_ulp_diffs;
  });
}
