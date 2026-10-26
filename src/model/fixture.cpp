#include "model/fixture.hpp"

#include <cstring>
#include <map>

#include "marincpp/core.hpp"
#include "model/npy.hpp"

namespace nm::model {

std::vector<float> load_fixture_storage(const std::string& dir, const std::vector<ParamSpec>& params) {
  int64_t total = 0;
  for (const ParamSpec& p : params) total += p.rows * p.cols;
  std::vector<float> out(static_cast<size_t>(total));
  struct File { std::vector<float> v; std::vector<int64_t> shape; };
  std::map<std::string, File> files;   // stacked layer files are read once
  int64_t base = 0;
  for (const ParamSpec& p : params) {
    for (const LeafSlice& leaf : p.leaves) {
      std::string file = leaf.name;
      int layer = -1;
      const std::string pre = "transformer.layers.";
      if (file.rfind(pre, 0) == 0) {
        const size_t dot = file.find('.', pre.size());
        layer = std::stoi(file.substr(pre.size(), dot - pre.size()));
        file = pre + file.substr(dot + 1);
      }
      auto it = files.find(file);
      if (it == files.end()) {
        File f;
        f.v = npy::load_f32(dir + "/" + file + ".npy", &f.shape);
        it = files.emplace(file, std::move(f)).first;
      }
      const File& f = it->second;
      int64_t leaf_rows = 0;
      for (size_t i = 0; i + 1 < leaf.rows.size(); i += 2) leaf_rows += leaf.rows[i + 1] - leaf.rows[i];
      const int64_t per = leaf_rows * p.cols;                 // elements of this leaf in one layer
      int64_t src = 0;
      if (layer >= 0) {
        NM_REQUIRE(!f.shape.empty() && f.shape[0] > layer, file + ": no layer " + std::to_string(layer));
        const int64_t layer_elems = static_cast<int64_t>(f.v.size()) / f.shape[0];
        NM_REQUIRE(layer_elems == per, file + ": " + std::to_string(layer_elems) + " elements per layer, storage " +
                                           p.name + " leaf expects " + std::to_string(per));
        src = int64_t(layer) * layer_elems;
      } else {
        NM_REQUIRE(static_cast<int64_t>(f.v.size()) == per,
                   file + ": " + std::to_string(f.v.size()) + " elements, expected " + std::to_string(per));
      }
      for (size_t i = 0; i + 1 < leaf.rows.size(); i += 2) {
        const int64_t n = (leaf.rows[i + 1] - leaf.rows[i]) * p.cols;
        std::memcpy(out.data() + base + leaf.rows[i] * p.cols, f.v.data() + src, size_t(n) * 4);
        src += n;
      }
    }
    base += p.rows * p.cols;
  }
  return out;
}

}  // namespace nm::model
