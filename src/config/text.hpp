// Internal to src/config: canonical value text, JSON/TOML quoting, the TOML subset reader.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "marincpp/config.hpp"

namespace nm {

// Canonical text of a knob value (doubles: shortest round-trip form of std::to_chars).
std::string to_text(bool v);
std::string to_text(int v);
std::string to_text(int64_t v);
std::string to_text(double v);
std::string to_text(const std::string& v);
bool from_text(std::string_view t, bool& v);
bool from_text(std::string_view t, int& v);
bool from_text(std::string_view t, int64_t& v);
bool from_text(std::string_view t, double& v);
bool from_text(std::string_view t, std::string& v);
const char* knob_type_name(KnobType t);

// JSON string; also a valid TOML basic string (control characters and DEL as 4-digit u-escapes), so full_toml
// round-trips every value.
std::string json_quote(std::string_view s);

// TOML subset: `[section]` headers, `key = value` with bare or dotted keys, basic "..." strings (every TOML 1.0
// escape: b t n f r, quote, backslash, 4- and 8-digit u/U code points) and literal '...' strings, decimal integers and
// floats in TOML 1.0 syntax, true/false, # comments. Arrays, inline tables, multi-line strings, hex/octal/binary
// numbers, inf/nan, control characters inside strings, and redefined keys or tables are rejected.
struct TomlEntry {
  std::string key;        // full dotted key "section.name"
  enum Kind { String, Int, Float, Bool } kind;
  std::string text;       // unquoted string value, or the number / bool as written (underscores removed)
  int line;
};
std::vector<TomlEntry> parse_toml(const std::string& text, const std::string& origin);
std::vector<TomlEntry> parse_toml_file(const std::string& path);

}  // namespace nm
