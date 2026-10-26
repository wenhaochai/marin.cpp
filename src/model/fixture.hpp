// Levanter tensors (cache/levref/<size>/<fixture>/{p0,g0,u0,...}/<leaf>.npy: f32, layers stacked on axis 0, each leaf
// row-major [out, in] after flattening its levanter axes) to and from marincpp's storage layout (baseline_params).
// Used by init.mode=fixture (parity runs start from the levanter init until T1.1 passes bitwise) and by the S7 gates.
//
// Mapping: a leaf "transformer.layers.<l>.<rest>" lives in file "transformer.layers.<rest>.npy" at index l of axis 0;
// any other leaf in "<leaf>.npy". The leaf's rows, in order, fill its LeafSlice row ranges of the storage tensor.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "marincpp/model.hpp"

namespace nm::model {

// All parameters of `params` (storage order, offsets = running sums of rows * cols) from the fixture directory `dir`
// (e.g. cache/levref/130m/step0_bf16/p0). Fatal if a file or a shape does not match.
std::vector<float> load_fixture_storage(const std::string& dir, const std::vector<ParamSpec>& params);

}  // namespace nm::model
