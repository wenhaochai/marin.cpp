#include "data/json.hpp"

#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "marincpp/core.hpp"

namespace nm::json {
namespace {

struct Parser {
  const std::string& s;
  const std::string& origin;
  size_t i = 0;

  [[noreturn]] void bad(const std::string& msg) {
    fatal(NM_WHERE, origin + ":" + std::to_string(i) + ": malformed JSON: " + msg);
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  bool lit(const char* w) {
    size_t n = std::char_traits<char>::length(w);
    if (s.compare(i, n, w) == 0) {
      i += n;
      return true;
    }
    return false;
  }
  static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) {
      o += char(cp);
    } else if (cp < 0x800) {
      o += char(0xC0 | (cp >> 6));
      o += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      o += char(0xE0 | (cp >> 12));
      o += char(0x80 | ((cp >> 6) & 0x3F));
      o += char(0x80 | (cp & 0x3F));
    } else {
      o += char(0xF0 | (cp >> 18));
      o += char(0x80 | ((cp >> 12) & 0x3F));
      o += char(0x80 | ((cp >> 6) & 0x3F));
      o += char(0x80 | (cp & 0x3F));
    }
  }
  uint32_t hex4() {
    if (i + 4 > s.size()) bad("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else bad("bad hex digit");
    }
    return v;
  }
  std::string str() {
    if (s[i] != '"') bad("expected a string");
    ++i;
    std::string o;
    while (true) {
      if (i >= s.size()) bad("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (static_cast<unsigned char>(c) < 0x20) bad("control character in string");
      if (c != '\\') {
        o += c;
        continue;
      }
      if (i >= s.size()) bad("unterminated escape");
      char e = s[i++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00) {
            if (!lit("\\u")) bad("lone high surrogate");
            uint32_t lo = hex4();
            if (lo < 0xDC00 || lo >= 0xE000) bad("bad low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp < 0xE000) {
            bad("lone low surrogate");
          }
          put_utf8(o, cp);
          break;
        }
        default: bad("bad escape");
      }
    }
    return o;
  }
  Value value(int depth) {
    if (depth > 64) bad("nesting deeper than 64");
    ws();
    if (i >= s.size()) bad("unexpected end");
    Value v;
    char c = s[i];
    if (c == '{') {
      v.kind = Value::Object;
      ++i;
      ws();
      if (i < s.size() && s[i] == '}') {
        ++i;
        return v;
      }
      while (true) {
        ws();
        if (i >= s.size()) bad("unexpected end in object");
        std::string k = str();
        ws();
        if (i >= s.size() || s[i] != ':') bad("expected ':'");
        ++i;
        v.obj[k] = value(depth + 1);
        ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == '}') {
          ++i;
          return v;
        }
        bad("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.kind = Value::Array;
      ++i;
      ws();
      if (i < s.size() && s[i] == ']') {
        ++i;
        return v;
      }
      while (true) {
        v.arr.push_back(value(depth + 1));
        ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == ']') {
          ++i;
          return v;
        }
        bad("expected ',' or ']'");
      }
    }
    if (c == '"') {
      v.kind = Value::String;
      v.text = str();
      return v;
    }
    if (lit("true")) {
      v.kind = Value::Bool;
      v.b = true;
      return v;
    }
    if (lit("false")) {
      v.kind = Value::Bool;
      return v;
    }
    if (lit("null")) return v;
    size_t b = i;
    if (i < s.size() && s[i] == '-') ++i;
    while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' || s[i] == 'E' ||
                            s[i] == '+' || s[i] == '-'))
      ++i;
    if (i == b) bad("unexpected character");
    v.kind = Value::Number;
    v.text = s.substr(b, i - b);
    return v;
  }
};

[[noreturn]] void type_error(const Value& v, const char* want) {
  fatal(NM_WHERE, std::string("JSON value is not ") + want + " (kind " + std::to_string(int(v.kind)) + ")");
}

}  // namespace

const Value& Value::at(const std::string& k) const {
  if (kind != Object) type_error(*this, "an object");
  auto it = obj.find(k);
  if (it == obj.end()) fatal(NM_WHERE, "JSON object has no key '" + k + "'");
  return it->second;
}

const Value& Value::at(size_t i) const {
  if (kind != Array) type_error(*this, "an array");
  if (i >= arr.size()) fatal(NM_WHERE, "JSON array index " + std::to_string(i) + " out of range");
  return arr[i];
}

uint64_t Value::as_u64() const {
  if (kind != Number) type_error(*this, "a number");
  uint64_t v = 0;
  auto r = std::from_chars(text.data(), text.data() + text.size(), v);
  if (r.ec != std::errc() || r.ptr != text.data() + text.size()) fatal(NM_WHERE, "JSON number '" + text + "' is not a uint64");
  return v;
}

int64_t Value::as_i64() const {
  if (kind != Number) type_error(*this, "a number");
  int64_t v = 0;
  auto r = std::from_chars(text.data(), text.data() + text.size(), v);
  if (r.ec != std::errc() || r.ptr != text.data() + text.size()) fatal(NM_WHERE, "JSON number '" + text + "' is not an int64");
  return v;
}

double Value::as_f64() const {
  if (kind != Number) type_error(*this, "a number");
  char* end = nullptr;
  double v = std::strtod(text.c_str(), &end);
  if (end != text.c_str() + text.size()) fatal(NM_WHERE, "JSON number '" + text + "' is malformed");
  return v;
}

const std::string& Value::as_str() const {
  if (kind != String) type_error(*this, "a string");
  return text;
}

bool Value::as_bool() const {
  if (kind != Bool) type_error(*this, "a bool");
  return b;
}

Value parse(const std::string& text, const std::string& origin) {
  Parser p{text, origin};
  Value v = p.value(0);
  p.ws();
  if (p.i != text.size()) p.bad("trailing characters");
  return v;
}

Value parse_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fatal(NM_WHERE, "cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return parse(ss.str(), path);
}

}  // namespace nm::json
