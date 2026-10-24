// Runner of nmtest.hpp.
#include "nmtest.hpp"

#include <exception>
#include <fstream>
#include <sstream>

namespace nmtest {
namespace {
std::string g_src = ".";
int g_fail_in_case = 0;
}  // namespace

std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

void fail(const char* file, int line, const std::string& what) {
  ++g_fail_in_case;
  std::printf("    FAIL %s:%d: %s\n", file, line, what.c_str());
}

const std::string& src_dir() { return g_src; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool file_exists(const std::string& path) { return std::ifstream(path).good(); }

}  // namespace nmtest

int main(int argc, char** argv) {
  std::string suite;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--src" && i + 1 < argc) nmtest::g_src = argv[++i];
    else suite = a;
  }
  int run = 0, failed = 0, skipped = 0;
  for (const nmtest::Case& c : nmtest::registry()) {
    if (!suite.empty() && suite != c.suite) continue;
    ++run;
    nmtest::g_fail_in_case = 0;
    std::string status = "PASS", note;
    try {
      c.fn();
    } catch (const nmtest::Skip& s) {
      status = "SKIP";
      note = s.msg;
    } catch (const std::exception& e) {
      ++nmtest::g_fail_in_case;
      note = std::string("uncaught exception: ") + e.what();
    }
    if (nmtest::g_fail_in_case) status = "FAIL";   // a case that failed and then skipped is a failure
    if (status == "FAIL") ++failed;
    if (status == "SKIP") ++skipped;
    std::printf("%s %s.%s%s%s\n", status.c_str(), c.suite, c.name, note.empty() ? "" : ": ", note.c_str());
  }
  std::printf("%s: %d cases, %d failed, %d skipped\n", suite.c_str(), run, failed, skipped);
  if (run == 0) return 1;
  if (failed) return 1;
  return skipped == run ? 77 : 0;
}
