// Minimal test harness for the C++ gate tests (no external dependency).
//   NM_TEST(suite, name) { CHECK(...); CHECK_EQ(a, b); CHECK_THROWS(expr, "substring"); SKIP("why"); }
// Run: nm_tests_s0 <suite> --src <repo root>. Exit 0 = every case passed, 77 = every case skipped (ctest
// SKIP_RETURN_CODE), 1 = a failure. A case with any failed check is FAIL, even if it skips afterwards.
#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace nmtest {

struct Case {
  const char* suite;
  const char* name;
  void (*fn)();
};
std::vector<Case>& registry();
struct Reg {
  Reg(const char* s, const char* n, void (*f)()) { registry().push_back({s, n, f}); }
};
struct Skip {
  std::string msg;
};
void fail(const char* file, int line, const std::string& what);
const std::string& src_dir();   // --src: the repository root (docs/, scripts/, cache/)
std::string read_file(const std::string& path);   // "" if absent
bool file_exists(const std::string& path);

template <class A, class B>
std::string show_pair(const A& a, const B& b) {
  std::ostringstream o;
  o.precision(17);
  o << a << " vs " << b;
  return o.str();
}

}  // namespace nmtest

#define NM_TEST(suite, name)                                                    \
  static void suite##__##name();                                                \
  static nmtest::Reg suite##__##name##_reg(#suite, #name, suite##__##name);     \
  static void suite##__##name()

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) nmtest::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_EQ(a, b)                                                                                    \
  do {                                                                                                    \
    const auto& nm_a_ = (a);                                                                              \
    const auto& nm_b_ = (b);                                                                              \
    if (!(nm_a_ == nm_b_))                                                                                \
      nmtest::fail(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b "): " + nmtest::show_pair(nm_a_, nm_b_)); \
  } while (0)

#define CHECK_THROWS(expr, substr)                                                                       \
  do {                                                                                                   \
    bool nm_thrown_ = false;                                                                             \
    try {                                                                                                \
      (void)(expr);                                                                                      \
    } catch (const std::exception& nm_e_) {                                                              \
      nm_thrown_ = true;                                                                                 \
      if (std::string(nm_e_.what()).find(substr) == std::string::npos)                                   \
        nmtest::fail(__FILE__, __LINE__,                                                                 \
                     std::string("exception of " #expr " lacks '") + substr + "': " + nm_e_.what());     \
    }                                                                                                    \
    if (!nm_thrown_) nmtest::fail(__FILE__, __LINE__, "no exception from " #expr);                      \
  } while (0)

#define SKIP(msg) throw nmtest::Skip{msg}
