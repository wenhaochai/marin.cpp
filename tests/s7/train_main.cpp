// S7 trainer driver: the multi-GPU step loop on the baseline (one graph per rank). For the first parity runs and
// throughput measurements; the real entry point `marincpp-train train` takes a config file (S10/S11).
//
//   nm_s7_train --root <scratch root> [--size 130m] [--gpus 4] [--steps 20] [--graph 1] [--micro-rows 32]
//               [--ce-chunk 32768] [--embed f32|bf16] [--attn fa3|ref] [--log-every 10] [--metrics out.jsonl]
#include <cstdio>
#include <cstring>
#include <string>

#include "train/trainer.hpp"

namespace {
std::string arg(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  return def;
}
}  // namespace

int main(int argc, char** argv) {
  using namespace nm;
  const std::string root = arg(argc, argv, "--root", "");
  if (root.empty()) {
    std::fprintf(stderr, "usage: nm_s7_train --root <scratch root> [options]\n");
    return 1;
  }
  train::TrainOptions o;
  o.size = arg(argc, argv, "--size", "130m");
  o.gpus = std::stoi(arg(argc, argv, "--gpus", "4"));
  o.steps = std::stoll(arg(argc, argv, "--steps", "20"));
  o.graph = arg(argc, argv, "--graph", "1") == "1";
  o.micro_rows = std::stoi(arg(argc, argv, "--micro-rows", "32"));
  o.ce_chunk = std::stoi(arg(argc, argv, "--ce-chunk", "32768"));
  o.embed = arg(argc, argv, "--embed", "f32") == "bf16" ? EmbedAccum::Bf16Sequential : EmbedAccum::F32;
  o.log_every = std::stoi(arg(argc, argv, "--log-every", "10"));
  o.attn = arg(argc, argv, "--attn", "fa3") == "ref" ? AttnBackendKind::RefF32 : AttnBackendKind::FA3;
  o.data_dir = root + "/marin_store_big/marincpp/fineweb-edu-10B-marin128k";
  o.init_fixture = root + "/project/marin.cpp/cache/levref/" + o.size + "/step0_bf16/p0";
  o.gemm_cache = arg(argc, argv, "--gemm-cache", (root + "/tmp/nm_gemm_" + o.size + ".bin").c_str());
  o.metrics = arg(argc, argv, "--metrics", "");
  try {
    return train::train(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 2;
  }
}
