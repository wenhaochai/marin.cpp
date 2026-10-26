// Minimal reader of NumPy .npy files (format 1.0 / 2.0, little-endian, C order): the levanter fixtures of
// cache/levref (f32 parameters and gradients, int32 token batches) that the S7 gates compare against. Test and tool
// code only; the training binary reads no .npy.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nm::npy {

struct Array {
  std::string descr;              // "<f4", "<i4", "<i8", ...
  std::vector<int64_t> shape;
  std::vector<char> data;         // raw little-endian payload
  int64_t numel() const;
  template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
};

Array load(const std::string& path);                 // fatal on a malformed file or a Fortran-order array
std::vector<float> load_f32(const std::string& path, std::vector<int64_t>* shape = nullptr);   // requires <f4
std::vector<int32_t> load_i32(const std::string& path, std::vector<int64_t>* shape = nullptr); // <i4, or <i8 narrowed

}  // namespace nm::npy
