// Extension registries: the one place a block, input layer, head or loss name exists (DESIGN.md §9.1). Knob checks
// call registry_has(); model.hpp register_* lines install factories.
#include <map>
#include <mutex>
#include <sstream>

#include "marincpp/config.hpp"
#include "marincpp/model.hpp"

namespace nm {
namespace {

struct Registry {
  std::mutex mu;
  std::map<std::string, void*> by_kind[4];
  Registry() {
    // Stage-1 baseline names; src/model (S7) installs their factories. See config.hpp registry_add.
    by_kind[static_cast<int>(RegistryKind::Block)]["sandwich"] = nullptr;
    by_kind[static_cast<int>(RegistryKind::Input)]["embed"] = nullptr;
    by_kind[static_cast<int>(RegistryKind::Head)]["full"] = nullptr;
  }
};

Registry& reg() {
  static Registry r;  // function-local: safe from static-initialization order of register_* lines
  return r;
}

const char* kind_name(RegistryKind k) {
  switch (k) {
    case RegistryKind::Block: return "block";
    case RegistryKind::Input: return "input";
    case RegistryKind::Head: return "head";
    case RegistryKind::Loss: return "loss";
  }
  return "?";
}

}  // namespace

void registry_add(RegistryKind k, const std::string& name, void* factory) {
  NM_REQUIRE(!name.empty() && name.find(',') == std::string::npos, "registry names are non-empty, without commas");
  Registry& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  auto& m = r.by_kind[static_cast<int>(k)];
  auto it = m.find(name);
  if (it != m.end() && it->second != nullptr)
    fatal(NM_WHERE, std::string("duplicate ") + kind_name(k) + " registration: " + name);
  m[name] = factory;
}

void* registry_factory(RegistryKind k, const std::string& name) {
  Registry& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  auto& m = r.by_kind[static_cast<int>(k)];
  auto it = m.find(name);
  return it == m.end() ? nullptr : it->second;
}

bool registry_has(RegistryKind k, const std::string& name) {
  Registry& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  return r.by_kind[static_cast<int>(k)].count(name) > 0;
}

std::vector<std::string> registry_names(RegistryKind k) {
  Registry& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  std::vector<std::string> out;
  for (auto& [n, f] : r.by_kind[static_cast<int>(k)]) out.push_back(n);
  return out;
}

bool registry_has_all(RegistryKind k, const std::string& comma_list) {
  if (comma_list.empty()) return true;
  std::stringstream ss(comma_list);
  std::string item;
  while (std::getline(ss, item, ','))
    if (item.empty() || !registry_has(k, item)) return false;
  return comma_list.back() != ',';
}

// model.hpp registration entry points.
void register_block(const char* name, BlockFactory f) { registry_add(RegistryKind::Block, name, reinterpret_cast<void*>(f)); }
void register_input(const char* name, InputFactory f) { registry_add(RegistryKind::Input, name, reinterpret_cast<void*>(f)); }
void register_head(const char* name, HeadFactory f) { registry_add(RegistryKind::Head, name, reinterpret_cast<void*>(f)); }
void register_loss(const char* name, LossFactory f) { registry_add(RegistryKind::Loss, name, reinterpret_cast<void*>(f)); }

std::unique_ptr<Block> make_block(const Config& c, int layer) {
  void* f = registry_factory(RegistryKind::Block, c.model_block);
  if (!f) fatal(NM_WHERE, "block '" + c.model_block + "' has no implementation in this build");
  return reinterpret_cast<BlockFactory>(f)(c, layer);
}

}  // namespace nm
