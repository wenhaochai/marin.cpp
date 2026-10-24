// TOML subset reader (text.hpp). Config files are flat knob tables, so this subset is the whole grammar they need;
// anything else is an error with file:line. Inside the subset the rules are TOML 1.0's: number syntax (no leading zeros,
// '_' only between digits), string escapes, no control characters in strings, every key and table defined once.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

#include "config/text.hpp"

namespace nm {
namespace {

[[noreturn]] void bad(const std::string& origin, int line, const std::string& msg) {
  fatal(NM_WHERE, origin + ":" + std::to_string(line) + ": " + msg);
}

bool bare_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

void skip_ws(std::string_view s, size_t& i) {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
}

// Dotted bare key starting at i.
std::string read_key(std::string_view s, size_t& i, const std::string& origin, int line) {
  std::string key;
  while (true) {
    skip_ws(s, i);
    size_t b = i;
    while (i < s.size() && bare_char(s[i])) ++i;
    if (i == b) bad(origin, line, "expected a bare key");
    key.append(s.substr(b, i - b));
    skip_ws(s, i);
    if (i < s.size() && s[i] == '.') {
      key.push_back('.');
      ++i;
      continue;
    }
    return key;
  }
}

void expect_end(std::string_view s, size_t i, const std::string& origin, int line) {
  skip_ws(s, i);
  if (i < s.size() && s[i] != '#') bad(origin, line, "unexpected text after the value: " + std::string(s.substr(i)));
}

// TOML forbids control characters other than tab inside strings (U+0000-U+0008, U+000A-U+001F, U+007F).
bool control_char(char c) {
  const unsigned char u = static_cast<unsigned char>(c);
  return (u < 0x20 && u != '\t') || u == 0x7f;
}

void put_utf8(std::string& o, uint32_t cp) {
  if (cp < 0x80) {
    o.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    o.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    o.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    o.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    o.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// TOML 1.0 decimal integers and floats (hex/octal/binary, inf and nan are not accepted by this subset).
const std::regex& int_re() {
  static const std::regex r(R"([+-]?(0|[1-9](_?[0-9])*))");
  return r;
}
const std::regex& float_re() {
  // integer part, then a fraction with an optional exponent, or an exponent alone (exponent digits may lead with 0)
  static const std::regex r(R"([+-]?(0|[1-9](_?[0-9])*))"
                            R"(((\.[0-9](_?[0-9])*)([eE][+-]?[0-9](_?[0-9])*)?|[eE][+-]?[0-9](_?[0-9])*))");
  return r;
}

}  // namespace

std::string json_quote(std::string_view s) {
  std::string o = "\"";
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      case '\r': o += "\\r"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned char>(c));
          o += b;
        } else {
          o += c;
        }
    }
  }
  return o + "\"";
}

std::vector<TomlEntry> parse_toml(const std::string& text, const std::string& origin) {
  std::vector<TomlEntry> out;
  // TOML 1.0 defines every key and every table once: `keys` holds the full keys of values, `tables` the [headers],
  // `dotted` the tables a dotted key created (they cannot be reopened by a header).
  std::set<std::string> keys, tables, dotted;
  std::string section;
  std::istringstream in(text);
  std::string raw;
  int line = 0;
  while (std::getline(in, raw)) {
    ++line;
    if (!raw.empty() && raw.back() == '\r') raw.pop_back();
    std::string_view s = raw;
    size_t i = 0;
    skip_ws(s, i);
    if (i == s.size() || s[i] == '#') continue;
    if (s[i] == '[') {
      ++i;
      if (i < s.size() && s[i] == '[') bad(origin, line, "arrays of tables are not supported");
      section = read_key(s, i, origin, line);
      if (i >= s.size() || s[i] != ']') bad(origin, line, "expected ']'");
      expect_end(s, i + 1, origin, line);
      if (keys.count(section)) bad(origin, line, "table [" + section + "] redefines the value " + section);
      if (dotted.count(section)) bad(origin, line, "table [" + section + "] was already defined by dotted keys");
      if (!tables.insert(section).second) bad(origin, line, "duplicate table [" + section + "]");
      continue;
    }
    std::string key = read_key(s, i, origin, line);
    if (i >= s.size() || s[i] != '=') bad(origin, line, "expected '=' after key " + key);
    ++i;
    skip_ws(s, i);
    if (i >= s.size()) bad(origin, line, "missing value for " + key);
    TomlEntry e;
    e.key = section.empty() ? key : section + "." + key;
    e.line = line;
    char c = s[i];
    if (c == '"') {
      if (s.substr(i, 3) == "\"\"\"") bad(origin, line, "multi-line strings are not supported");
      ++i;
      std::string v;
      while (true) {
        if (i >= s.size()) bad(origin, line, "unterminated string");
        char ch = s[i++];
        if (ch == '"') break;
        if (control_char(ch)) bad(origin, line, "control character in a string (write it as an escape)");
        if (ch != '\\') {
          v.push_back(ch);
          continue;
        }
        if (i >= s.size()) bad(origin, line, "unterminated escape");
        char esc = s[i++];
        switch (esc) {
          case '"': v.push_back('"'); break;
          case '\\': v.push_back('\\'); break;
          case 'b': v.push_back('\b'); break;
          case 't': v.push_back('\t'); break;
          case 'n': v.push_back('\n'); break;
          case 'f': v.push_back('\f'); break;
          case 'r': v.push_back('\r'); break;
          case 'u':
          case 'U': {
            const size_t n = esc == 'u' ? 4 : 8;
            if (i + n > s.size()) bad(origin, line, std::string("short \\") + esc + " escape");
            uint32_t cp = 0;
            for (size_t k = 0; k < n; ++k) {
              const char h = s[i + k];
              const int d = (h >= '0' && h <= '9') ? h - '0'
                            : (h >= 'a' && h <= 'f') ? h - 'a' + 10
                            : (h >= 'A' && h <= 'F') ? h - 'A' + 10
                                                     : -1;
              if (d < 0) bad(origin, line, std::string("bad hex digit in \\") + esc + " escape");
              cp = cp * 16 + static_cast<uint32_t>(d);
            }
            i += n;
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
              bad(origin, line, "escape is not a Unicode scalar value");
            put_utf8(v, cp);
            break;
          }
          default: bad(origin, line, std::string("unsupported escape \\") + esc);
        }
      }
      e.kind = TomlEntry::String;
      e.text = v;
    } else if (c == '\'') {
      if (s.substr(i, 3) == "'''") bad(origin, line, "multi-line strings are not supported");
      size_t j = s.find('\'', i + 1);
      if (j == std::string_view::npos) bad(origin, line, "unterminated literal string");
      e.kind = TomlEntry::String;
      e.text = std::string(s.substr(i + 1, j - i - 1));
      for (char ch : e.text)
        if (control_char(ch)) bad(origin, line, "control character in a literal string");
      i = j + 1;
    } else if (c == '[' || c == '{') {
      bad(origin, line, "arrays and inline tables are not supported");
    } else {
      size_t b = i;
      while (i < s.size() && s[i] != ' ' && s[i] != '\t' && s[i] != '#') ++i;
      std::string tok(s.substr(b, i - b));
      if (tok == "true" || tok == "false") {
        e.kind = TomlEntry::Bool;
        e.text = tok;
      } else {
        std::string bare = (tok.size() && (tok[0] == '+' || tok[0] == '-')) ? tok.substr(1) : tok;
        if (bare == "inf" || bare == "nan") bad(origin, line, "non-finite numbers are not accepted: " + tok);
        const bool is_int = std::regex_match(tok, int_re());
        if (!is_int && !std::regex_match(tok, float_re()))
          bad(origin, line, "malformed value: " + tok + " (TOML decimal integer or float)");
        std::string num;
        for (char ch : tok)
          if (ch != '_') num.push_back(ch);
        double d;
        int64_t n;
        if (is_int ? !from_text(num, n) : !from_text(num, d)) bad(origin, line, "value out of range: " + tok);
        e.kind = is_int ? TomlEntry::Int : TomlEntry::Float;
        e.text = num;
      }
    }
    expect_end(s, i, origin, line);
    if (keys.count(e.key)) bad(origin, line, "duplicate key " + e.key);
    if (tables.count(e.key) || dotted.count(e.key)) bad(origin, line, "key " + e.key + " redefines a table");
    for (size_t dot = e.key.find('.'); dot != std::string::npos; dot = e.key.find('.', dot + 1)) {
      const std::string prefix = e.key.substr(0, dot);
      if (keys.count(prefix)) bad(origin, line, "key " + e.key + " extends the value " + prefix);
      if (prefix.size() > section.size()) dotted.insert(prefix);   // a table created by this dotted key
    }
    keys.insert(e.key);
    out.push_back(std::move(e));
  }
  return out;
}

std::vector<TomlEntry> parse_toml_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fatal(NM_WHERE, "cannot open config file " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse_toml(ss.str(), path);
}

}  // namespace nm
