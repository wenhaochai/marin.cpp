// MemoryPlan: declaration, layout, printing (memory.hpp; DESIGN.md §2.1-§2.2).
#include <algorithm>
#include <cinttypes>
#include <map>

#include "marincpp/memory.hpp"

namespace nm {

int64_t BufferSpec::numel() const {
  if (ndim <= 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < ndim; ++i)
    if (__builtin_mul_overflow(n, shape[i], &n)) return INT64_MAX;
  return n;
}

uint64_t BufferSpec::bytes() const {
  const int64_t n = numel();
  if (n <= 0) return 0;
  uint64_t b = 0;
  if (__builtin_mul_overflow(static_cast<uint64_t>(n), static_cast<uint64_t>(dtype_size(dtype)), &b))
    return kMaxBufferBytes;
  return b < kMaxBufferBytes ? b : kMaxBufferBytes;
}
static_assert(kCanaryBytes % kArenaAlign == 0, "guard bands keep buffer offsets aligned");

const char* lifetime_name(Lifetime l) {
  switch (l) {
    case Lifetime::Persistent: return "persistent";
    case Lifetime::Microbatch: return "microbatch";
    case Lifetime::Activation: return "activation";
    case Lifetime::Transient: return "transient";
    case Lifetime::Workspace: return "workspace";
  }
  return "?";
}

char lifetime_class(Lifetime l) {
  switch (l) {
    case Lifetime::Persistent: return 'P';
    case Lifetime::Microbatch:
    case Lifetime::Activation: return 'A';
    case Lifetime::Transient: return 'T';
    case Lifetime::Workspace: return 'W';
  }
  return '?';
}

BufferId MemoryPlan::add(BufferSpec spec) {
  NM_REQUIRE(!finalized_, "MemoryPlan::add after finalize");
  NM_REQUIRE(!spec.name.empty(), "buffer without a name");
  NM_REQUIRE(spec.ndim >= 1 && spec.ndim <= 4, "buffer " + spec.name + ": ndim must be 1..4");
  for (int i = 0; i < spec.ndim; ++i) NM_REQUIRE(spec.shape[i] > 0, "buffer " + spec.name + ": empty dimension");
  if (spec.group.empty()) spec.group = spec.name;
  auto [it, inserted] = index_.emplace(spec.name, static_cast<int32_t>(specs_.size()));
  if (!inserted) fatal(NM_WHERE, "duplicate buffer name " + spec.name);
  specs_.push_back(std::move(spec));
  return BufferId{it->second};
}

BufferId MemoryPlan::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? BufferId{} : BufferId{it->second};
}

void MemoryPlan::finalize(uint64_t budget_bytes, bool canaries) {
  canaries_ = canaries;
  offsets_.assign(specs_.size(), 0);
  auto align = [](uint64_t x) { return (x + kArenaAlign - 1) / kArenaAlign * kArenaAlign; };
  // Placement order: Persistent, Microbatch, Activation (by layer), Transient, Workspace; declaration order inside.
  std::vector<int32_t> order(specs_.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int32_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    const BufferSpec &x = specs_[a], &y = specs_[b];
    if (x.life != y.life) return x.life < y.life;
    if (x.life == Lifetime::Activation) return x.layer < y.layer;
    return false;
  });
  uint64_t dev = 0, host = 0;
  for (int32_t i : order) {
    const bool guarded = canaries && !specs_[i].host_pinned;
    uint64_t& cur = specs_[i].host_pinned ? host : dev;
    cur = align(cur);
    if (guarded) cur += kCanaryBytes;   // leading band [offset - 4 KiB, offset); kCanaryBytes keeps the alignment
    offsets_[i] = cur;
    cur += specs_[i].bytes();
    if (guarded) cur += kCanaryBytes;   // trailing band [offset + bytes, offset + bytes + 4 KiB)
  }
  device_bytes_ = align(dev);
  pinned_bytes_ = align(host);
  finalized_ = true;
  if (device_bytes_ > budget_bytes)
    fatal(NM_WHERE, "memory plan needs " + std::to_string(device_bytes_) + " B per GPU, budget " +
                        std::to_string(budget_bytes) + " B");
}

uint64_t MemoryPlan::device_bytes() const {
  NM_REQUIRE(finalized_, "plan not finalized");
  return device_bytes_;
}
uint64_t MemoryPlan::pinned_bytes() const {
  NM_REQUIRE(finalized_, "plan not finalized");
  return pinned_bytes_;
}
uint64_t MemoryPlan::payload_bytes() const {
  uint64_t s = 0;
  for (auto& b : specs_)
    if (!b.host_pinned) s += b.bytes();
  return s;
}
uint64_t MemoryPlan::payload_bytes(char cls) const {
  uint64_t s = 0;
  for (auto& b : specs_)
    if (!b.host_pinned && lifetime_class(b.life) == cls) s += b.bytes();
  return s;
}
uint64_t MemoryPlan::offset(BufferId id) const {
  NM_REQUIRE(finalized_ && id.valid() && id.idx < size(), "offset of an invalid buffer or unfinalized plan");
  return offsets_[id.idx];
}
const BufferSpec& MemoryPlan::spec(BufferId id) const {
  NM_REQUIRE(id.valid() && id.idx < size(), "invalid buffer id");
  return specs_[id.idx];
}

std::vector<MemoryPlan::Group> MemoryPlan::groups() const {
  std::vector<Group> out;
  std::map<std::string, size_t> at;
  for (auto& b : specs_) {
    if (b.host_pinned) continue;
    auto it = at.find(b.group);
    if (it == at.end()) {
      at[b.group] = out.size();
      out.push_back(Group{b.group, b.life, 1, b.bytes()});
    } else {
      out[it->second].count += 1;
      out[it->second].bytes += b.bytes();
    }
  }
  return out;
}

static std::string shape_text(const BufferSpec& b) {
  std::string s = "[";
  for (int i = 0; i < b.ndim; ++i) s += (i ? ", " : "") + std::to_string(b.shape[i]);
  return s + "]";
}

static void print_totals(const MemoryPlan& p, FILE* f) {
  const double GiB = 1024.0 * 1024.0 * 1024.0;
  std::fprintf(f, "subtotals GiB: P %.2f  A %.2f  T %.2f  W %.2f\n", p.payload_bytes('P') / GiB,
               p.payload_bytes('A') / GiB, p.payload_bytes('T') / GiB, p.payload_bytes('W') / GiB);
  std::fprintf(f, "payload %" PRIu64 " B (%.2f GiB)", p.payload_bytes(), p.payload_bytes() / GiB);
  if (p.finalized())
    std::fprintf(f, "; arena %" PRIu64 " B (alignment%s)", p.device_bytes(), p.canaries() ? " + guard bands" : "");
  std::fprintf(f, "\n");
}

void MemoryPlan::print(FILE* f) const {
  std::fprintf(f, "%-40s %-10s %-5s %-24s %16s %16s\n", "buffer", "life", "dtype", "shape", "bytes", "offset");
  for (int i = 0; i < size(); ++i) {
    const BufferSpec& b = specs_[i];
    std::fprintf(f, "%-40s %-10s %-5s %-24s %16" PRIu64 " %16" PRIu64 "\n", b.name.c_str(), lifetime_name(b.life),
                 dtype_name(b.dtype), shape_text(b).c_str(), b.bytes(), finalized_ ? offsets_[i] : 0);
  }
  print_totals(*this, f);
}

void MemoryPlan::print_grouped(FILE* f) const {
  std::fprintf(f, "| %-36s | %c | %5s | %16s |\n", "buffer", 'L', "count", "bytes total");
  for (auto& g : groups())
    std::fprintf(f, "| %-36s | %c | %5d | %16" PRIu64 " |\n", g.name.c_str(), lifetime_class(g.life), g.count, g.bytes);
  print_totals(*this, f);
}

}  // namespace nm
