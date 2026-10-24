// Minimal JSON reader for the converter's meta.json and the test references (RFC 8259 subset: no duplicate-key check).
// Numbers keep their text, so 64-bit integers are exact (as_u64 / as_i64 parse the text, never a double).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nm::json {

struct Value {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  std::string text;                        // Number: as written; String: decoded UTF-8
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  bool has(const std::string& k) const { return kind == Object && obj.count(k); }
  const Value& at(const std::string& k) const;   // fatal if absent or not an object
  const Value& at(size_t i) const;               // fatal if out of range or not an array
  uint64_t as_u64() const;
  int64_t as_i64() const;
  double as_f64() const;
  const std::string& as_str() const;
  bool as_bool() const;
};

Value parse(const std::string& text, const std::string& origin);   // fatal with origin:offset on malformed input
Value parse_file(const std::string& path);

}  // namespace nm::json
