// Model graph and the plug-in interfaces for research variants: blocks, input layers, output heads, readouts, loss
// terms, depth state, step regimes, RNG and statistics hooks. DESIGN.md §3 (kernel schedule) and §9.3 (variants).
//
// A variant is C++ code that (1) declares its parameters, activations, transients, depth use and statistic slots,
// (2) records its forward and its hand-written backward on a stream, and (3) ships its gates (DESIGN.md §10): the
// existing JAX code as the reference (tools/levref variant mode), V1-V4. The model never synchronizes with the host;
// everything is recorded into the step graph of each regime.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "marincpp/config.hpp"
#include "marincpp/core.hpp"
#include "marincpp/memory.hpp"

namespace nm {

struct DeviceContext;
class AttentionBackend;
class GemmCache;
class Arena;
class GradReducer;

// ------------------------------------------------------------------------------------------------ parameters
// Optimizer groups are data (optim.hpp GroupSpec); a parameter names its group. Baseline names: "muonh", "adamh",
// "adam"; variants add their own (e.g. "depth_adam": hc/mhc/attnres parameters with Adam eps 1e-8).
// Scope of the norm-preserving projection: per layer matrix (MuonH), per whole tensor (AdamH lm_head), or per block of
// a RowPartition (optim.hpp; per-row control without the whole-table confound, VO Q5b).
enum class NormScope : uint8_t { LayerMatrix = 0, WholeTensor = 1, RowPartition = 2 };

// Init (model.md §4): truncated normal (-3, 3) via the threefry2x32 port plus ports of XLA's f32 Erf and ErfInv
// (jax 0.11.1), uniform f*(b-a)+a with the contraction XLA's PTX shows, not re-standardized, times `scale_f32`.
// scale_f32 is computed in f64 and rounded once: (float)(1.0 / sqrt((double)fan_in)); the draw is multiplied by it
// (t * f32(1/sqrt(D))), never divided. SameDrawAs reuses another parameter's draw with its own scale (lm_head and
// embedding share k_emb). Until T1.1 passes bitwise, parity runs use init.mode=fixture.
enum class InitKind : uint8_t { TruncNormal = 0, Ones = 1, Zeros = 2, Constant = 3, SameDrawAs = 4 };

struct InitSpec {
  InitKind kind = InitKind::TruncNormal;
  float scale_f32 = 1.0f;
  std::vector<uint32_t> key_path;     // indices into the split tree of model.md §4.2, e.g. {2 (model), 0 (k_t), l, 0, 1}
  std::string same_draw_as;
  float value = 0.0f;                 // Constant
};

// A logical leaf inside a storage tensor. q, k, v live in one fused Wqkv buffer and gate/up in one Wgu buffer (rows
// interleaved in 64-row blocks), but each logical leaf keeps its own levanter name, its own MuonH matrix and its own
// grad/params norm. Slices end on leaf boundaries so per-leaf norms are exact sums of slice partials.
struct LeafSlice {
  std::string name;                   // levanter leaf: "transformer.layers.3.self_attn.k_proj.weight"
  std::vector<int64_t> rows;          // storage rows belonging to this leaf, as [begin, end) pairs
};

struct ParamSpec {
  std::string name;                   // storage name: "L3.Wqkv"
  std::vector<int64_t> shape;         // row-major storage
  int layer = -1;
  std::string group = "adam";         // GroupSpec name
  NormScope scope = NormScope::WholeTensor;
  int row_partition = -1;             // index of a RowPartition (optim.hpp), NormScope::RowPartition
  int64_t rows = 0, cols = 0;         // [OUT, IN] matrix view (MuonH / AdamH)
  std::vector<LeafSlice> leaves;      // one entry with all rows for unfused tensors
  InitSpec init;
  bool vocab_sized = false;           // embedding / heads: row-sharded under ZeRO-1, candidates for ZeRO-2
};

// ------------------------------------------------------------------------------------------------ statistics hooks
// Per-layer statistics without host syncs: each stat owns slots in a device float array; kernels write per-CTA
// partials that a fixed-order reduction turns into slot values; a cross-rank reduction follows; the metrics ring
// copies the slots. Cadence is a regime (DESIGN.md §2.4), not a runtime test inside the graph.
enum class HookPoint : uint8_t { ResidIn, AttnOut, MlpOut, ResidOut, ResidGrad, ParamGrad, Param, ReadoutState,
                                 ReadoutGrad, LossTok, Custom };
enum class StatKind : uint8_t {
  Rms = 0, AbsMax = 1, SumSq = 2, Mean = 3,
  Dot = 4,           // <a, b> and ||a||, ||b|| (pls probe cosines: readout trunk gradient vs final-loss gradient)
  Vector = 5,        // `width` values (attnres source weights per sublayer)
  PosRangeMean = 6,  // mean of loss_tok over positions [pos_begin, pos_end) (ICL score = late minus early CE)
};
enum class StatReduce : uint8_t { Sum = 0, Max = 1, PerRank = 2 };   // across ranks (worst-rank loss: Max / PerRank)
struct StatSpec {
  std::string key;                    // "stats/L3/resid_rms", "grad/norm/transformer.layers.3.mlp.up_proj.weight"
  int layer = -1;
  HookPoint point = HookPoint::Custom;
  HookPoint point_b = HookPoint::Custom;   // second operand of Dot
  StatKind kind = StatKind::Rms;
  StatReduce reduce = StatReduce::Sum;
  int width = 1;                      // Vector
  int pos_begin = 0, pos_end = 0;     // PosRangeMean
  std::string regime = "stats";       // regime whose graph computes it ("stats" = k % 10 == 0 && k > 1)
  int slot = -1;                      // assigned at plan time
};

// ------------------------------------------------------------------------------------------------ declarations
struct ActSpec {
  std::string name;                   // "x_in", "qkv", "gu" (planner prefixes act.L<l>.)
  DType dtype = DType::bf16;
  int64_t per_token = 0;              // elements per token
  Recompute recompute = Recompute::Never;
  Lifetime life = Lifetime::Activation;   // Activation, Transient or Microbatch
};

// Depth state (attnres, attnres_block, moda): sources visible to later sublayers within one microbatch. Declared once
// per model; lives in the Microbatch region. attnres at d48: 2L+1 = 97 sources x T x D bf16 (1.6 GB at T = 16,384)
// plus their gradient accumulators (grad_dtype; the gradient of source i sums over every later sublayer).
struct DepthDecl {
  int n_sources = 0;
  int64_t per_token = 0;              // elements per token per source (D)
  DType dtype = DType::bf16;
  DType grad_dtype = DType::f32;
  bool per_source_rstd = true;        // f32 1/RMS per source and token
};
struct DepthUse {
  int writes_source = -1;             // index this sublayer's output is stored at (-1 none)
  int n_visible = 0;                  // sources [0, n_visible) this sublayer reads
};

struct BlockDecl {
  std::vector<ParamSpec> params;
  std::vector<ActSpec> acts;          // kept forward -> backward
  std::vector<ActSpec> transients;    // scratch inside the layer
  std::vector<StatSpec> stats;
  int streams = 1;                    // residual stream copies (hc/mhc: m); the stream buffer is [T, streams * D]
  std::vector<DepthUse> depth;        // one per sublayer (2 for attention + MLP blocks); empty = no depth state
};

// Pointers of one microbatch (all inside the arena; fixed for the life of the graphs).
struct Microbatch {
  int index = 0;                      // 0 .. n_micro-1
  int64_t T = 0;                      // tokens
  int rows = 0;
  const int32_t* tokens = nullptr;    // [rows, S]
  const int32_t* targets = nullptr;
  const float* weight = nullptr;
  const int32_t* pos = nullptr;
  const int32_t* cu_seqlens = nullptr;   // packed segment starts (varlen path)
  const int32_t* n_segs = nullptr;       // device count (read by the persistent backward scheduler)
  const int32_t* row_seg_start = nullptr;  // [rows, max_segs_per_seq + 1] per-row segment table (doc_causal path)
};

// Device-side schedules every kernel may read with the device step k (never a host value baked into the graph).
struct DeviceSchedules {
  const float* lr = nullptr;          // main eta(k)
  const float* adam_lr = nullptr;
  const float* bc1 = nullptr;
  const float* bc2 = nullptr;
  const float* const* custom = nullptr;  // variant tables (readout weights, lr multipliers, noise scales), by index
  int n_custom = 0;
};

// Device threefry2x32 key; kernels fold it by (k, rank, microbatch, layer, purpose) so draws differ per device and
// microbatch and are reproducible. Gate: device bits equal host bits (data.hpp threefry2x32), bitwise.
struct RngKey { uint32_t k0, k1; };
enum class RngPurpose : uint32_t { Dropout = 1, InputNoise = 2, SampledNegatives = 3, Custom = 16 };

struct LayerCtx {
  DeviceContext* dev = nullptr;
  cudaStream_t stream = nullptr;
  int layer = 0;
  const ModelShape* m = nullptr;
  const ExecShape* e = nullptr;
  Numerics num;
  const Microbatch* mb = nullptr;
  AttentionBackend* attn = nullptr;
  GemmCache* gemm = nullptr;
  const bf16_t* stream_in = nullptr;  // residual stream entering the layer [T, streams*D]
  bf16_t* stream_out = nullptr;
  const bf16_t* d_stream_out = nullptr;  // residual gradient arriving from above (backward)
  bf16_t* d_stream_in = nullptr;
  // depth state (DepthDecl): sources[i] is [T, D]; d_sources[i] accumulates the gradient of source i
  const bf16_t* const* sources = nullptr;
  void* const* d_sources = nullptr;   // DepthDecl::grad_dtype
  const float* const* source_rstd = nullptr;
  int n_visible = 0;
  float* stat_partials = nullptr;     // this layer's statistic partial slots
  bool last_microbatch = false;       // backward: the last accumulation step (gradient buckets become ready)
  // step state (research needs: dropout from pass p, input noise, train-only behaviour, step-gated masks)
  bool training = true;               // false in the eval graphs
  const int32_t* step = nullptr;      // device k
  const DeviceSchedules* sched = nullptr;
  RngKey rng{0, 0};
  uint64_t regime_flags = 0;          // static flags of the regime this graph was captured for
};

// Binding by declared name with dtype/shape checks (memory.hpp Arena::bind); the block prefixes its own layer.
struct BindCtx {
  const Arena* arena = nullptr;
  int layer = -1;
  void* get(const std::string& local_name, DType dtype, std::initializer_list<int64_t> shape) const;
  float* grad(const std::string& param_name) const;       // f32 gradient slice of a declared parameter
  bf16_t* param16(const std::string& param_name) const;   // replicated bf16 compute copy
};

// ------------------------------------------------------------------------------------------------ interfaces
class Block {
 public:
  virtual ~Block() = default;
  virtual void declare(int layer, const ModelShape& m, BlockDecl& d) const = 0;
  virtual void bind(int layer, const BindCtx& b) = 0;
  virtual void forward(const LayerCtx& c) = 0;
  // Hand-written backward. Accumulates (+=) into the f32 parameter gradients; writes d_stream_in; adds into d_sources.
  virtual void backward(const LayerCtx& c) = 0;
  // remat=min: recompute the Recompute::Elementwise activations; remat=layer: recompute every activation.
  virtual void recompute(const LayerCtx& c, Remat policy) { (void)c; (void)policy; }
  // The state a readout after this layer reads (DESIGN.md §9.3). Default: the residual stream itself (no copy;
  // `out` may alias stream_out). hc/mhc: sum of streams; attnres: the next layer's attention mix.
  virtual const bf16_t* readout_state(const LayerCtx& c, bf16_t* scratch) { (void)scratch; return c.stream_out; }
  // Adds the readout's gradient w.r.t. that state into this layer's residual / depth gradients.
  virtual void readout_state_bwd(const LayerCtx& c, const bf16_t* d_state);
};

// Tokens -> x0 and its backward. Default "embed": bf16 gather + sorted segment sum (kernels/elementwise.hpp). Others:
// composed rows (cv/cvd/cvr), hashed n-gram tables (ShardedTable, comm.hpp), byte inputs, input noise.
class InputLayer {
 public:
  virtual ~InputLayer() = default;
  virtual void declare(const ModelShape& m, BlockDecl& d) const = 0;
  virtual void bind(const BindCtx& b) = 0;
  virtual void forward(const LayerCtx& c, bf16_t* x0) = 0;
  virtual void backward(const LayerCtx& c, const bf16_t* dx0) = 0;
  // Rows touched in this step (lazy embedding updates, per-row control); nullptr if not tracked.
  virtual const uint8_t* touched_rows() const { return nullptr; }
};

// What an output head exposes to loss terms and statistics.
struct HeadOutputs {
  const float* loss_tok = nullptr;    // [T] per-token loss (0 where weight 0)
  const float* lse = nullptr;         // [T]
  const float* weight = nullptr;      // [T] label mask after the head's own masking
  float* loss_sum = nullptr;          // device scalar partial (fixed-order reduced)
};

// Final hidden -> per-token loss, dx, its own dW. Default "full": the fused linear CE of kernels/cross_entropy.hpp.
// Others: "sampled" (candidates = sort-unique targets of the device microbatch + golden-ratio stride negatives,
// remapped targets, gathered W_P, fused CE, unique-row scatter of dW_P; memory planned for P_max; overflow fallback by a
// conditional node), composed heads, logit bias, label masks (rm), focal, a second OD head.
class OutputHead {
 public:
  virtual ~OutputHead() = default;
  virtual void declare(const ModelShape& m, BlockDecl& d) const = 0;
  virtual void bind(const BindCtx& b) = 0;
  // Forward and gradient in one call (the gradient is computed in the forward pass). dx == nullptr: no dX GEMM.
  virtual void forward_backward(const LayerCtx& c, const bf16_t* x, bf16_t* dx, bool accumulate_dx) = 0;
  virtual void forward_only(const LayerCtx& c, const bf16_t* x, float* loss_tok) = 0;   // eval
  virtual HeadOutputs outputs() const = 0;
};

// Registries; names are the knob values. Knob checks call registry_has() (declared in config.hpp with RegistryKind),
// so a name exists in one place: its register_* line.
using BlockFactory = std::unique_ptr<Block> (*)(const Config& c, int layer);
using InputFactory = std::unique_ptr<InputLayer> (*)(const Config& c);
using HeadFactory = std::unique_ptr<OutputHead> (*)(const Config& c, int readout);
void register_block(const char* name, BlockFactory f);
void register_input(const char* name, InputFactory f);
void register_head(const char* name, HeadFactory f);
std::unique_ptr<Block> make_block(const Config& c, int layer);   // looks up model.block

// Built-in blocks (src/model/blocks/*.cu): sandwich (the baseline: hybrid norm), preln, lns, deepnorm, keel, hc, mhc,
// attnres, attnres_block. moda registers only when the FA3 extra_kv patch exists (attention.hpp).

// ------------------------------------------------------------------------------------------------ readout table
// Several readouts per layer (pls: separate heads, probes, monitors over one trunk). Each has a mode, its own head
// (or a shared one), its own optimizer group, and a weight schedule (device table). Probe and monitor readouts and
// detached backbones have a static `train_backbone == false`: the head runs with dx = nullptr (no dX GEMM; the readout
// norm backward computes only its gain gradient). Late start (readout.on) is a regime boundary, not a device scalar.
enum class ReadoutMode : uint8_t { Train = 0, Probe = 1, Monitor = 2 };
struct ReadoutSpec {
  int layer;                          // readout state after this layer (L-1 = the final head)
  ReadoutMode mode = ReadoutMode::Train;
  std::string head = "full";          // OutputHead registry name
  bool own_norm = false, own_head = false;
  std::string group;                  // optimizer group of its head and norm
  int weight_table = -1;              // DeviceSchedules::custom index (readout.weight * shape factor, readout.off)
  bool train_head = true;             // false: detach_head (shared)
  bool train_backbone = true;         // static per regime
  bool local = false;                 // backbone gradient stops below `layer`
  int visible_dims = 0;
};
std::vector<ReadoutSpec> readout_table(const Config& c, const ModelShape& m);

// Auxiliary loss terms (MTP, z-loss, ...): own parameters, own forward/backward, own metric key. They read the main
// head's outputs (lse, loss_tok, label mask) and add into d(final hidden).
class LossTerm {
 public:
  virtual ~LossTerm() = default;
  virtual const char* key() const = 0;   // "train/aux/<name>"
  virtual void declare(const ModelShape& m, BlockDecl& d) const = 0;
  virtual void bind(const BindCtx& b) = 0;
  virtual void forward_backward(const LayerCtx& c, const HeadOutputs& head, bf16_t* d_final_hidden) = 0;
};
using LossFactory = std::unique_ptr<LossTerm> (*)(const Config& c);
void register_loss(const char* name, LossFactory f);

// ------------------------------------------------------------------------------------------------ step regimes
// A regime is a step range with static flags known on the host; each distinct regime is one captured g_step in the
// same arena (the plan is the maximum over regimes). The host picks the graph from k before each launch.
enum RegimeFlag : uint64_t {
  kRegimeStats = 1ull << 0,           // params/grad norms + hook stats (k % 10 == 0 && k > 1)
  kRegimeDropout = 1ull << 1,         // dropout / input noise active (DROP_FROM_PASS)
  kRegimeReadoutsOff = 1ull << 2,     // readout.off reached: readouts skipped
  kRegimeBackboneOn = 1ull << 3,      // readout.on reached: readouts send dX
  kRegimeSampled = 1ull << 4,         // sampled softmax stage (P in `param`)
  kRegimeDiagnostic = 1ull << 5,      // record_backward per-term gradients
  kRegimeFreeze = 1ull << 6,          // OV_FREEZE_AT and similar step-gated freezes
};
struct StepRegime {
  int64_t step_begin = 0, step_end = 0;   // [begin, end)
  uint64_t flags = 0;
  int64_t param = 0;                  // e.g. sampled-softmax P, phase index of a DataPlan (batch / seq_len change)
  std::string name;
};
// Splits [0, N) into regimes from the config (stats cadence, readout schedules, data phases, variant schedules).
std::vector<StepRegime> step_regimes(const Config& c, int64_t n_steps);

// ------------------------------------------------------------------------------------------------ the model
class Model {
 public:
  explicit Model(const Config& c);
  // All parameter specs in the canonical order: input layer, layers 0..L-1 (block order), final norm, output head,
  // readouts. The gradient buffer is laid out in REVERSE of this order per layer, so buckets are contiguous and become
  // ready in backward order (comm.hpp).
  const std::vector<ParamSpec>& params() const;
  void declare(MemoryPlan& plan, const ExecShape& e);  // parameters, depth state, activations, transients, stats
  void bind(const Arena& arena);
  // Records forward + backward of one microbatch for one regime.
  void record_microbatch(DeviceContext& d, const Microbatch& mb, bool last, GradReducer& reducer,
                         const StepRegime& regime);
  // Diagnostic regime: per-term gradients (term_mask over main loss, readouts, aux losses) into grad_out slots;
  // costs one extra backward per selected term.
  void record_backward(DeviceContext& d, const Microbatch& mb, uint64_t term_mask, float* const* grad_out);
  // Forward only (eval): per-position loss of every readout requested.
  void record_eval_batch(DeviceContext& d, const Microbatch& mb, float* const* loss_tok_per_readout);
  void init_parameters(DeviceContext& d);   // threefry port, fixture, or init.from mapping (checkpoint.hpp); bf16 casts
  const std::vector<StatSpec>& stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nm
