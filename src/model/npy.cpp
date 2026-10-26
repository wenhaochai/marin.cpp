#include "model/npy.hpp"

#include <cstring>
#include <fstream>

#include "marincpp/core.hpp"

namespace nm::npy {

int64_t Array::numel() const {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

namespace {

// The value of `key` in the header dict, as raw text up to the next top-level ',' or '}'.
std::string field(const std::string& h, const std::string& key, const std::string& path) {
  const size_t k = h.find("'" + key + "'");
  if (k == std::string::npos) fatal(NM_WHERE, path + ": npy header has no '" + key + "'");
  size_t i = h.find(':', k) + 1;
  while (i < h.size() && h[i] == ' ') ++i;
  int depth = 0;
  size_t j = i;
  for (; j < h.size(); ++j) {
    const char c = h[j];
    if (c == '(' || c == '[') ++depth;
    if (c == ')' || c == ']') --depth;
    if (depth == 0 && (c == ',' || c == '}')) break;
  }
  return h.substr(i, j - i);
}

}  // namespace

Array load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fatal(NM_WHERE, "cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) fatal(NM_WHERE, path + ": not an npy file");
  const int major = static_cast<unsigned char>(magic[6]);
  uint32_t hlen = 0;
  if (major == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string h(hlen, '\0');
  f.read(h.data(), hlen);
  if (!f) fatal(NM_WHERE, path + ": truncated npy header");
  Array a;
  std::string d = field(h, "descr", path);
  a.descr = d.substr(1, d.size() - 2);
  if (field(h, "fortran_order", path).find("False") == std::string::npos)
    fatal(NM_WHERE, path + ": Fortran-order arrays are not supported");
  const std::string sh = field(h, "shape", path);
  for (size_t i = 0; i < sh.size();) {
    while (i < sh.size() && (sh[i] < '0' || sh[i] > '9')) ++i;
    if (i >= sh.size()) break;
    size_t j = i;
    while (j < sh.size() && sh[j] >= '0' && sh[j] <= '9') ++j;
    a.shape.push_back(std::stoll(sh.substr(i, j - i)));
    i = j;
  }
  size_t item = 0;
  if (a.descr == "<f4" || a.descr == "<i4" || a.descr == "<u4") item = 4;
  else if (a.descr == "<f8" || a.descr == "<i8") item = 8;
  else if (a.descr == "<u2" || a.descr == "<f2") item = 2;
  else if (a.descr == "|u1" || a.descr == "|b1") item = 1;
  else fatal(NM_WHERE, path + ": unsupported dtype " + a.descr);
  a.data.resize(static_cast<size_t>(a.numel()) * item);
  f.read(a.data.data(), static_cast<std::streamsize>(a.data.size()));
  if (!f) fatal(NM_WHERE, path + ": truncated npy payload");
  return a;
}

std::vector<float> load_f32(const std::string& path, std::vector<int64_t>* shape) {
  Array a = load(path);
  if (a.descr != "<f4") fatal(NM_WHERE, path + ": expected <f4, got " + a.descr);
  if (shape) *shape = a.shape;
  std::vector<float> v(static_cast<size_t>(a.numel()));
  std::memcpy(v.data(), a.data.data(), a.data.size());
  return v;
}

std::vector<int32_t> load_i32(const std::string& path, std::vector<int64_t>* shape) {
  Array a = load(path);
  if (shape) *shape = a.shape;
  std::vector<int32_t> v(static_cast<size_t>(a.numel()));
  if (a.descr == "<i4") {
    std::memcpy(v.data(), a.data.data(), a.data.size());
  } else if (a.descr == "<i8") {
    const int64_t* p = a.as<int64_t>();
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int32_t>(p[i]);
  } else {
    fatal(NM_WHERE, path + ": expected <i4 or <i8, got " + a.descr);
  }
  return v;
}

}  // namespace nm::npy
