// The knob table, generated from knobs.def (DESIGN.md §9.1): parser, CLI, hash, label and `knobs` all read it.
#include <charconv>
#include <cmath>
#include <string>
#include <system_error>

#include "config/text.hpp"
#include "marincpp/config.hpp"

namespace nm {

std::string to_text(bool v) { return v ? "true" : "false"; }
std::string to_text(int v) { return std::to_string(v); }
std::string to_text(int64_t v) { return std::to_string(v); }
std::string to_text(double v) {
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof(buf), v);  // shortest text that round-trips
  return std::string(buf, r.ptr);
}
std::string to_text(const std::string& v) { return v; }

bool from_text(std::string_view t, bool& v) {
  if (t == "true") { v = true; return true; }
  if (t == "false") { v = false; return true; }
  return false;
}
template <class I>
static bool parse_int(std::string_view t, I& v) {
  if (!t.empty() && t[0] == '+') t.remove_prefix(1);
  I x{};
  auto r = std::from_chars(t.data(), t.data() + t.size(), x);
  if (r.ec != std::errc() || r.ptr != t.data() + t.size() || t.empty()) return false;
  v = x;
  return true;
}
bool from_text(std::string_view t, int& v) { return parse_int(t, v); }
bool from_text(std::string_view t, int64_t& v) { return parse_int(t, v); }
bool from_text(std::string_view t, double& v) {
  if (!t.empty() && t[0] == '+') t.remove_prefix(1);
  double x = 0;
  auto r = std::from_chars(t.data(), t.data() + t.size(), x);
  if (r.ec != std::errc() || r.ptr != t.data() + t.size() || t.empty() || !std::isfinite(x)) return false;
  v = x;
  return true;
}
bool from_text(std::string_view t, std::string& v) {
  v = std::string(t);
  return true;
}

namespace {
template <class T> constexpr KnobType type_of();
template <> constexpr KnobType type_of<bool>() { return KnobType::Bool; }
template <> constexpr KnobType type_of<int>() { return KnobType::Int; }
template <> constexpr KnobType type_of<int64_t>() { return KnobType::Int64; }
template <> constexpr KnobType type_of<double>() { return KnobType::Double; }
template <> constexpr KnobType type_of<std::string>() { return KnobType::String; }

const Config& defaults() {
  static const Config d;
  return d;
}

// Label tags (knobs.def): at most one conversion, and it matches the knob type; default_label() substitutes the value
// itself, so a tag is never used as a printf format.
template <class T> consteval char tag_conversion();
template <> consteval char tag_conversion<bool>() { return 0; }
template <> consteval char tag_conversion<int>() { return 'd'; }
template <> consteval char tag_conversion<int64_t>() { return 'd'; }
template <> consteval char tag_conversion<double>() { return 'g'; }
template <> consteval char tag_conversion<std::string>() { return 's'; }
consteval bool tag_ok(const char* t, char want) {
  int n = 0;
  for (const char* p = t; *p; ++p)
    if (*p == '%') {
      if (p[1] != want || want == 0) return false;
      ++n;
      ++p;
    }
  return n <= 1;
}
#define NM_KNOB(sec, name, type, def, flags, tag, check, doc) \
  static_assert(tag_ok(tag, tag_conversion<type>()), "knobs.def: label tag of " #sec "." #name " does not match its type");
#include "marincpp/knobs.def"
#undef NM_KNOB
}  // namespace

const std::vector<KnobInfo>& knob_table() {
  static const std::vector<KnobInfo> t = {
#define NM_KNOB(sec, name, type, def, flags, tag, check, doc)                                                   \
  KnobInfo{#sec "." #name, type_of<type>(), flags, tag, doc,                                                    \
           [](const Config& c) { return to_text(c.sec##_##name); },                                             \
           [](Config& c, std::string_view s) { return from_text(s, c.sec##_##name); },                          \
           [](const Config& c) {                                                                                \
             [[maybe_unused]] const auto& v = c.sec##_##name;                                                   \
             return static_cast<bool>(check);                                                                   \
           },                                                                                                   \
           [](const Config& c) { return c.sec##_##name == defaults().sec##_##name; }},
#include "marincpp/knobs.def"
#undef NM_KNOB
  };
  return t;
}

const KnobInfo* find_knob(std::string_view key) {
  for (const KnobInfo& k : knob_table())
    if (key == k.key) return &k;
  return nullptr;
}

const char* knob_type_name(KnobType t) {
  switch (t) {
    case KnobType::Bool: return "bool";
    case KnobType::Int: return "int";
    case KnobType::Int64: return "int64";
    case KnobType::Double: return "double";
    case KnobType::String: return "string";
  }
  return "?";
}

}  // namespace nm
