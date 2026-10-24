// s0_core: hashing, bf16 rounding, views, errors, registries.
#include <cmath>
#include <string>

#include <stdexcept>

#include "core/capi_util.hpp"
#include "marincpp/capi.h"
#include "marincpp/config.hpp"
#include "marincpp/core.hpp"
#include "marincpp/model.hpp"
#include "nmtest.hpp"

using namespace nm;

NM_TEST(s0_core, sha256_fips_vectors) {
  CHECK_EQ(sha256_hex(""), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK_EQ(sha256_hex("abc"), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  CHECK_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
           std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
  CHECK_EQ(sha256_hex(std::string(1000000, 'a')),
           std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
  // lengths around the 55/56/64-byte padding boundaries (references: Python hashlib)
  const std::pair<size_t, const char*> pad[] = {
      {55, "d5e285683cd4efc02d021a5c62014694958901005d6f71e89e0989fac77e4072"},
      {56, "04c26261370ee7541549d16dee320c723e3fd14671e66a099afe0a377c16888e"},
      {63, "75220b47218278e656f2013bb8f0c455a25eaf01e86c64924e9d48d89776d6f2"},
      {64, "7ce100971f64e7001e8fe5a51973ecdfe1ced42befe7ee8d5fd6219506b5393c"},
      {65, "9537c5fdf120482f7d58d25e9ed583f52c02b4e304ea814db1633ad565aed7e9"},
      {119, "000b48d4edf0fa7bee3c6236ecd2785baa5db4eeb8bb54341b029e0d9fa5fb0c"},
      {120, "13f05a0b594787f5ecd315edc96141bd3243203d1b7d4f0836f37308b276ba98"}};
  for (auto& [n, hex] : pad) CHECK_EQ(sha256_hex(std::string(n, 'x')), std::string(hex));
}

NM_TEST(s0_core, bf16_round_nearest_even) {
  CHECK_EQ(f32_to_bf16_bits(0.125f), uint16_t(0x3E00));
  CHECK_EQ(f32_to_bf16_bits(static_cast<float>(1.0 / std::sqrt(128.0))), uint16_t(0x3DB5));
  CHECK_EQ(bf16_bits_to_f32(0x3DB5), 0.08837890625f);
  CHECK_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F808000u)), uint16_t(0x3F80));  // tie -> even
  CHECK_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F818000u)), uint16_t(0x3F82));  // tie -> even
  CHECK_EQ(f32_to_bf16_bits(std::bit_cast<float>(0x3F808001u)), uint16_t(0x3F81));  // above the tie
  float nan = std::bit_cast<float>(0x7f800001u);
  uint16_t b = f32_to_bf16_bits(nan);
  CHECK((b & 0x7f80) == 0x7f80 && (b & 0x007f) != 0);
}

NM_TEST(s0_core, dtypes_and_views) {
  CHECK_EQ(dtype_size(DType::bf16), size_t(2));
  CHECK_EQ(dtype_size(DType::f64), size_t(8));
  CHECK_EQ(std::string(dtype_name(DType::u32)), std::string("u32"));
  TensorView v = make_view(nullptr, DType::f32, {3, 4, 5});
  CHECK_EQ(v.numel(), int64_t(60));
  CHECK_EQ(v.nbytes(), size_t(240));
  CHECK_EQ(v.stride[0], int64_t(20));
  CHECK_EQ(v.stride[2], int64_t(1));
}

NM_TEST(s0_core, fatal_is_catchable_with_location) {
  CHECK_THROWS(fatal(NM_WHERE, "boom"), "test_core.cpp");
  CHECK_THROWS(([] { NM_REQUIRE(1 == 2, "context"); })(), "requirement failed: 1 == 2: context");
  CHECK_THROWS(cuda_check(2, "cudaMalloc(&p, n)", NM_WHERE), "cudaErrorMemoryAllocation");
  CHECK_THROWS(nccl_check(5, "ncclAllReduce", NM_WHERE), "ncclInvalidUsage");
}

static std::unique_ptr<Block> null_block(const Config&, int) { return nullptr; }

NM_TEST(s0_core, registries) {
  CHECK(registry_has(RegistryKind::Block, "sandwich"));
  CHECK(registry_has(RegistryKind::Input, "embed"));
  CHECK(registry_has(RegistryKind::Head, "full"));
  CHECK(!registry_has(RegistryKind::Block, "moda"));   // not registered until its FA3 extra_kv kernel exists
  CHECK(registry_has_all(RegistryKind::Loss, ""));
  CHECK(!registry_has_all(RegistryKind::Loss, "zloss"));
  CHECK(registry_factory(RegistryKind::Block, "sandwich") == nullptr);   // installed by src/model (S7)
  Config c;
  CHECK_THROWS(make_block(c, 0), "has no implementation");
  register_block("s0_test_block", &null_block);
  CHECK(registry_has(RegistryKind::Block, "s0_test_block"));
  CHECK_THROWS(register_block("s0_test_block", &null_block), "duplicate block registration");
  CHECK(registry_has_all(RegistryKind::Block, "sandwich,s0_test_block"));
  CHECK(!registry_has_all(RegistryKind::Block, "sandwich,"));
}

NM_TEST(s0_core, capi_status_codes) {
  // A caller can tell a CUDA/NCCL/cuBLAS failure from a bad argument (capi.h).
  CHECK_EQ(capi_call([] {}), NM_OK);
  CHECK_EQ(std::string(nm_last_error()), std::string(""));
  CHECK_EQ(capi_call([] { NM_CUDA(1); }), NM_ERR_DEVICE);   // cudaErrorInvalidValue
  CHECK(std::string(nm_last_error()).find("cudaErrorInvalidValue") != std::string::npos);
  CHECK_EQ(capi_call([] { NM_NCCL(4); }), NM_ERR_DEVICE);
  CHECK_EQ(capi_call([] { NM_CUBLAS(7); }), NM_ERR_DEVICE);
  CHECK_EQ(capi_call([] { NM_REQUIRE(false, "bad rows"); }), NM_ERR_INVALID);
  CHECK(std::string(nm_last_error()).find("bad rows") != std::string::npos);
  CHECK_EQ(capi_call([] { throw std::runtime_error("other"); }), NM_ERR_INTERNAL);
  CHECK_EQ(capi_call([] {}), NM_OK);
  CHECK_EQ(std::string(nm_last_error()), std::string(""));   // cleared by the next successful call
}

// Not part of s0_core: ctest s0_runner_selftest expects this case to be reported FAIL (a failure is never hidden by a
// later SKIP).
NM_TEST(s0_runner_selftest, fail_then_skip) {
  nmtest::fail(__FILE__, __LINE__, "intended failure (runner self-test)");
  SKIP("a skip after a failure");
}
