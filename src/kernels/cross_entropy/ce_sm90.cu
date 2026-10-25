// sm_90a kernels of the fused CE (WGMMA + TMA, warp-specialized, persistent). Built for NM_ARCHS_SM90A only; the
// dispatchers of ce_api.cpp call them on compute capability 9.0 and fall back to the portable kernels when they return
// false (a shape or store dtype they do not take).
//
// k_ce_logits_sm90 (DESIGN.md §3.8 step 1): z = x W^T for 128-row x 128-column tiles, K = D in blocks of 64.
//   warpgroup 0: producer (one thread issues the TMA loads of x and W tiles into a ring of S stages)
//   warpgroups 1, 2: consumers in ping-pong: the CTA's tiles alternate between them, the stage ring is filled in tile
//     order, and an order barrier (named barriers 3, 4) starts a warpgroup's mainloop only after the other one has
//     waited on every stage of the previous tile: one mainloop runs while the other warpgroup does its epilogue, and no
//     consumer waits on a stage a full ring phase ahead (needs D / 64 >= S). Each consumer issues two
//     wgmma.m64n128k16 per k16 step (rows 0-63, 64-127 of its tile) into 128 f32 accumulator registers per thread.
//   epilogue (from the f32 accumulators): per row the tile's (max, sum expf(z - max)) over its valid columns, z[target]
//     when the target falls in the tile, and the bf16 logits store through a 128B-swizzled shared-memory tile and two
//     TMA stores (columns past V and rows past `rows` are clipped by TMA).
// Tile order: groups of GM row blocks (GM * 128 * D * 2 B ~ 16 MB of x stays in L2), column blocks outer inside a group,
// so the CTAs in flight share W tiles and the x group.
#include <cuda.h>
#include <cudaTypedefs.h>

#include <algorithm>
#include <mutex>
#include <string>

#include "kernels/cross_entropy/ce_internal.hpp"
#include "kernels/cross_entropy/ce_mma.cuh"

namespace nm::ce {
namespace {

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          dst),
      "l"(map), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, uint32_t src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::"l"(map), "r"(c0),
               "r"(c1), "r"(src)
               : "memory");
}
__device__ __forceinline__ void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void tma_store_wait_read0() {
  asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void tma_store_wait0() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ uint4 ld_shared_v4(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}
__device__ __forceinline__ void st_shared_v4(uint32_t addr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ float ld_shared_f32(uint32_t addr) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t ld_shared_u32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}
__device__ __forceinline__ void st_shared_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory");
}
template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N));
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// Shared-memory matrix descriptor of a K-major bf16 tile with rows of 64 elements (128 B) and the 128B swizzle (the
// layout a TMA load with CU_TENSOR_MAP_SWIZZLE_128B writes); 8-row groups 1024 B apart. `addr` must be 1024-aligned
// for the tile, plus 32 B per k16 step inside the row.
__device__ __forceinline__ uint64_t desc_k128(uint32_t addr) {
  uint64_t d = 0;
  d |= uint64_t((addr & 0x3FFFF) >> 4);
  d |= uint64_t(1) << 16;           // leading byte offset (unused with the swizzle)
  d |= uint64_t(1024 >> 4) << 32;   // stride byte offset: 8 rows x 128 B
  d |= uint64_t(1) << 62;           // 128B swizzle
  return d;
}

// A descriptor from its two words: the low word (start address >> 4, leading byte offset) changes per stage and k
// step, the high word (stride byte offset, swizzle mode) is a constant. Offsets are added to the low word only: the
// start field (address >> 4 < 2^14) never carries into the leading-offset field.
__device__ __forceinline__ uint64_t desc_join(uint32_t lo, uint32_t hi) {
  uint64_t d;
  asm("mov.b64 %0, {%1, %2};\n" : "=l"(d) : "r"(lo), "r"(hi));
  return d;
}
__device__ __forceinline__ uint32_t desc_lo(uint64_t d) { return uint32_t(d); }
__device__ __forceinline__ uint32_t desc_hi(uint64_t d) { return uint32_t(d >> 32); }

template <class T>
__device__ __forceinline__ void fence_regs(T& r) {
  asm volatile("" : "+f"(r)::"memory");
}

// D[64 x 128] (+)= A[64 x 16] B[16 x 128], both K-major in shared memory; 64 f32 accumulators per thread.
__device__ __forceinline__ void wgmma_m64n128(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "%64, %65, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}

template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n256(float (&d)[128], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, "
      "%96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, "
      "%112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, "
      "%128, %129, p, 1, 1, %131, %132;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]),
        "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]),
        "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
        "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]),
        "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]),
        "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]),
        "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]),
        "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]),
        "+f"(d[126]), "+f"(d[127])
      : "l"(da), "l"(db), "r"(scale_d), "n"(TA), "n"(TB));
}

template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n192(float (&d)[96], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %98, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, "
      "%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, "
      "%48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, "
      "%64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, "
      "%80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, "
      "%96, %97, p, 1, 1, %99, %100;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]),
        "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]),
        "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
        "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]),
        "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
      : "l"(da), "l"(db), "r"(scale_d), "n"(TA), "n"(TB));
}

__device__ __forceinline__ void tma_load_1d(uint32_t dst, const CUtensorMap* map, int c0, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];\n" ::"r"(dst),
      "l"(map), "r"(c0), "r"(bar)
      : "memory");
}

// K-major bf16 tile with rows of 32 elements (64 B) and the 64B swizzle (TMA CU_TENSOR_MAP_SWIZZLE_64B); 8-row groups
// 512 B apart; +32 B per k16 step.
__device__ __forceinline__ uint64_t desc_k64(uint32_t addr) {
  uint64_t d = 0;
  d |= uint64_t((addr & 0x3FFFF) >> 4);
  d |= uint64_t(1) << 16;
  d |= uint64_t(512 >> 4) << 32;
  d |= uint64_t(2) << 62;           // 64B swizzle
  return d;
}

// MN-major bf16 operand in 128B-swizzled boxes [k rows][64 mn elements] (TMA box {64, k}): 8-k-row groups 1024 B apart
// (SBO), consecutive 64-wide mn blocks `mn_stride` bytes apart (LBO); +2048 B per k16 step (16 rows of 128 B).
// (bench/s5_ce/probe_mn.cu checks this assignment on the H100.)
constexpr uint32_t kMnLboIsBlockStride = 1;
__device__ __forceinline__ uint64_t desc_mn128(uint32_t addr, uint32_t mn_stride) {
  const uint32_t lbo = kMnLboIsBlockStride ? mn_stride : 1024, sbo = kMnLboIsBlockStride ? 1024 : mn_stride;
  uint64_t d = 0;
  d |= uint64_t((addr & 0x3FFFF) >> 4);
  d |= uint64_t((lbo >> 4) & 0x3FFF) << 16;
  d |= uint64_t((sbo >> 4) & 0x3FFF) << 32;
  d |= uint64_t(1) << 62;
  return d;
}

// ------------------------------------------------------------------------------------------------ logits kernel
constexpr int L9_BM = 128, L9_BN = 128, L9_BK = 64, L9_S = 4;
constexpr int L9_TILE_A = L9_BM * L9_BK * 2, L9_TILE_B = L9_BN * L9_BK * 2;   // 16 KB each
constexpr int L9_STAGE = L9_TILE_A + L9_TILE_B;
constexpr int L9_STG = L9_BM * L9_BN * 2;                                     // bf16 staging tile, 32 KB
constexpr int L9_SMEM = 1024 + L9_S * L9_STAGE + 2 * L9_STG + 2 * L9_S * 8;
static_assert(L9_BN == kTile, "one statistics tile per column block");

struct Logits9 {
  CUtensorMap tm_x, tm_w, tm_st;
  const int32_t* targets;
  float* tmax; float* tsum; float* zt;
  int64_t ldT;
  int rows, V, KB;
  int n_mb, n_nb, group_m;
  int write_store;
  int fast_exp;
};

__device__ __forceinline__ void tile_coords(int t, const Logits9& p, int& mb, int& nb) {
  const int per_group = p.group_m * p.n_nb;
  const int grp = t / per_group, rem = t - grp * per_group;
  const int gm0 = grp * p.group_m;
  const int gsz = min(p.group_m, p.n_mb - gm0);
  nb = rem / gsz;
  mb = gm0 + rem % gsz;
}

// v = (k == sel) ? a : v, opaque to the compiler (a select chain over register-array elements is otherwise turned into
// an indexed load, which moves the accumulators to local memory)
__device__ __forceinline__ void sel_if(float& v, float a, int k, int sel) {
  asm("{\n.reg .pred q;\nsetp.eq.s32 q, %2, %3;\nselp.f32 %0, %1, %0, q;\n}\n" : "+f"(v) : "f"(a), "r"(k), "r"(sel));
}

// Per row of the 128 x 128 tile (this thread: rows (h, r) = h * 64 + warp * 16 + g + 8 r, columns jn * 8 + 2 t4 + e):
// (max, sum expf(z - max)) over the valid columns, reduced over the 4 lanes of the row, and z[target] by the thread that
// holds the target column. FULL: every column is below V. The two rows r = 0, 1 of a half h are processed together and
// the exponentials in blocks of 4, with fixed-order add trees: one epilogue warp per scheduler and ~40 free registers
// leave no thread-level parallelism, so the latency of the libdevice expf chains must be hidden inside the thread.
template <bool FULL, bool FAST>
__device__ __forceinline__ void stats_rows(const float (&acc)[2][64], const Logits9& p, int nb, int m0, int n0, int warp,
                                           int g, int t4, const int (&tgt)[4]) {
  auto ok = [&](int jn, int e) { return FULL || n0 + jn * 8 + 2 * t4 + e < p.V; };
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float mx[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float m4[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
      for (int jn = 0; jn < 16; ++jn)
#pragma unroll
        for (int e = 0; e < 2; ++e)
          if (ok(jn, e)) m4[(jn & 1) * 2 + e] = fmaxf(m4[(jn & 1) * 2 + e], acc[h][jn * 4 + r * 2 + e]);
      float m = fmaxf(fmaxf(m4[0], m4[1]), fmaxf(m4[2], m4[3]));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
      mx[r] = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    }
    // branch-free: every exponential is evaluated (a conditional one becomes a branch per element, which serializes
    // them); an all -inf row uses 0 as its max (expf(-inf) = 0), invalid columns are selected away afterwards
    const float ms[2] = {mx[0] == -INFINITY ? 0.f : mx[0], mx[1] == -INFINITY ? 0.f : mx[1]};
    float sum[2] = {0.f, 0.f};
#pragma unroll
    for (int jb = 0; jb < 16; jb += 2) {   // columns of n8 blocks jb, jb + 1: 4 per row, both rows
      float ex[2][4];
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int jn = jb + (q >> 1), e = q & 1;
          const float x = exp_stat<FAST>(__fsub_rn(acc[h][jn * 4 + r * 2 + e], ms[r]));
          ex[r][q] = ok(jn, e) ? x : 0.f;
        }
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sum[r] = __fadd_rn(sum[r], __fadd_rn(__fadd_rn(ex[r][0], ex[r][1]), __fadd_rn(ex[r][2], ex[r][3])));
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float sr = __fadd_rn(sum[r], __shfl_xor_sync(0xffffffffu, sum[r], 1));
      sr = __fadd_rn(sr, __shfl_xor_sync(0xffffffffu, sr, 2));
      const int gm = m0 + h * 64 + warp * 16 + g + 8 * r;
      if (gm < p.rows) {
        if (t4 == 0) {
          p.tmax[int64_t(nb) * p.ldT + gm] = mx[r];
          p.tsum[int64_t(nb) * p.ldT + gm] = sr;
        }
        const int tc = tgt[h * 2 + r] - n0;
        if (unsigned(tc) < unsigned(L9_BN) && ((tc >> 1) & 3) == t4) {
          const int jsel = tc >> 3;
          float v = 0.f;
#pragma unroll
          for (int jn = 0; jn < 16; ++jn)
            sel_if(v, (tc & 1) ? acc[h][jn * 4 + r * 2 + 1] : acc[h][jn * 4 + r * 2], jn, jsel);
          p.zt[gm] = v;
        }
      }
    }
  }
}

__global__ void __launch_bounds__(384, 1) k_ce_logits_sm90(const __grid_constant__ Logits9 p) {
  extern __shared__ uint8_t smraw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
  const uint32_t sbase = smem_u32(sm);
  const uint32_t stg_base = sbase + L9_S * L9_STAGE;
  const uint32_t bar_full = stg_base + 2 * L9_STG, bar_empty = bar_full + L9_S * 8;
  const int wg = threadIdx.x >> 7;
  const int total = p.n_mb * p.n_nb;

  if (threadIdx.x == 0) {
    for (int s = 0; s < L9_S; ++s) {
      mbar_init(bar_full + 8 * s, 1);
      mbar_init(bar_empty + 8 * s, 4);   // one arrival per consumer warp
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (wg == 0) {
    setmaxnreg_dec<40>();   // 40 + 2 x 232 = 3 x 168: setmaxnreg.inc only takes what a .dec released
    if (threadIdx.x == 0) {
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&p.tm_x) : "memory");
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(&p.tm_w) : "memory");
      uint32_t it = 0;
      for (int t = blockIdx.x; t < total; t += gridDim.x) {
        int mb, nb;
        tile_coords(t, p, mb, nb);
        for (int kb = 0; kb < p.KB; ++kb, ++it) {
          const uint32_t s = it % L9_S, ph = (it / L9_S) & 1;
          mbar_wait(bar_empty + 8 * s, ph ^ 1);
          mbar_expect_tx(bar_full + 8 * s, L9_STAGE);
          const uint32_t a = sbase + s * L9_STAGE;
          tma_load_2d(a, &p.tm_x, kb * L9_BK, mb * L9_BM, bar_full + 8 * s);
          tma_load_2d(a + L9_TILE_A, &p.tm_w, kb * L9_BK, nb * L9_BN, bar_full + 8 * s);
        }
      }
    }
    return;
  }

  setmaxnreg_inc<232>();
  const int c = wg - 1;
  const int tw = threadIdx.x & 127, warp = tw >> 5, lane = tw & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const uint32_t stg = stg_base + c * L9_STG;
  const uint32_t a_lo = desc_lo(desc_k128(sbase)), a_hi = desc_hi(desc_k128(sbase));
  const uint32_t b_lo = desc_lo(desc_k128(sbase + L9_TILE_A)), b_hi = a_hi;
  uint32_t it = 0;
  int j = 0;
  for (int t = blockIdx.x; t < total; t += gridDim.x, ++j) {
    if ((j & 1) != c) {
      it += p.KB;
      continue;
    }
    int mb, nb;
    tile_coords(t, p, mb, nb);
    // ping-pong order: this warpgroup's mainloop starts after the other one finished waiting on every stage of the
    // previous tile, so no consumer waits on a stage more than one ring phase ahead (mbarrier parity is 1 bit)
    int tgt[4];   // targets of this thread's rows (h, r), loaded before the mainloop
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int gm = mb * L9_BM + (q >> 1) * 64 + warp * 16 + g + 8 * (q & 1);
      tgt[q] = gm < p.rows ? p.targets[gm] : -1;
    }
    if (j > 0) named_bar(3 + c, 256);
    float acc[2][64];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 64; ++i) acc[h][i] = 0.f;
    for (int kb = 0; kb < p.KB; ++kb, ++it) {
      const uint32_t s = it % L9_S, ph = (it / L9_S) & 1;
      mbar_wait(bar_full + 8 * s, ph);
      // descriptors = base + (byte offset >> 4) in the low word, built once, not per k-block (ncu: the per-k-block
      // construction and its R2UR moves were the most executed instructions of the mainloop)
      const uint32_t off = (s * L9_STAGE) >> 4;
      wgmma_fence();
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const uint64_t db = desc_join(b_lo + off + 2 * k, b_hi);
        wgmma_m64n128(acc[0], desc_join(a_lo + off + 2 * k, a_hi), db, 1);
        wgmma_m64n128(acc[1], desc_join(a_lo + off + (64 * 128 >> 4) + 2 * k, a_hi), db, 1);
      }
      wgmma_commit();
      wgmma_wait<1>();
      if (kb > 0 && lane == 0) mbar_arrive(bar_empty + 8 * ((it - 1) % L9_S));
    }
    if (t + int(gridDim.x) < total) named_arrive(3 + (c ^ 1), 256);   // the next tile's warpgroup may start
    wgmma_wait<0>();
    if (lane == 0) mbar_arrive(bar_empty + 8 * ((it - 1) % L9_S));
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int i = 0; i < 64; ++i) fence_regs(acc[h][i]);

    // ---- epilogue: statistics and z[target] from the f32 accumulators
    const int m0 = mb * L9_BM, n0 = nb * L9_BN;
    if (n0 + L9_BN <= p.V) {
      if (p.fast_exp) stats_rows<true, true>(acc, p, nb, m0, n0, warp, g, t4, tgt);
      else stats_rows<true, false>(acc, p, nb, m0, n0, warp, g, t4, tgt);
    } else {   // the last column block of a ragged V
      if (p.fast_exp) stats_rows<false, true>(acc, p, nb, m0, n0, warp, g, t4, tgt);
      else stats_rows<false, false>(acc, p, nb, m0, n0, warp, g, t4, tgt);
    }

    // ---- the bf16 logits store: swizzled staging tile [2 boxes of 128 rows x 64 columns], then two TMA stores
    if (p.write_store) {
      if (tw == 0) tma_store_wait_read0();   // this warpgroup's previous store has read the staging tile
      named_bar(1 + c, 128);
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int rl = h * 64 + warp * 16 + g + 8 * r;
#pragma unroll
          for (int jn = 0; jn < 16; ++jn) {
            const int box = jn >> 3, ch = jn & 7;
            const uint32_t off = box * (L9_BM * 128) + rl * 128 + ((ch ^ (rl & 7)) << 4) + t4 * 4;
            st_shared_u32(stg + off, pack_bf16x2(acc[h][jn * 4 + r * 2], acc[h][jn * 4 + r * 2 + 1]));
          }
        }
      fence_proxy_async();
      named_bar(1 + c, 128);
      if (tw == 0) {
        tma_store_2d(&p.tm_st, stg, n0, m0);
        tma_store_2d(&p.tm_st, stg + L9_BM * 128, n0 + 64, m0);
        tma_store_commit();
      }
    }
  }
  if (p.write_store && tw == 0) tma_store_wait0();
}

// ------------------------------------------------------------------------------------------------ dgrad / wgrad
// One kernel for both CE GEMMs (DESIGN.md §3.8 step 3, B1.5 / B2, and the plain-dz GEMMs of B1): 64 output rows per
// tile, N = 2 x WN columns of D per tile (WN = 256; 192 when D is a multiple of 384 but not of 512), K loop in blocks of
// 32 (V for dgrad, rows for wgrad), S-stage TMA ring.
//   KIND 0 dgrad: dx[rows, D] = dz[rows, V] W[V, D]; A = dz tiles [64 rows][32 v] (K-major, SW64); B = W tiles
//                 [32 v][64 d] (MN-major, SW128); bf16 output (levanter / accurate accumulate forms of DgradArgs)
//   KIND 1 wgrad: dw[V, D] (+)= dz^T x; A = dz^T from tiles [32 rows][64 v] (MN-major, SW128); B = x tiles
//                 [32 rows][64 d] (MN-major, SW128); f32 read-modify-write (+ f32(bf16(acc)) per chunk when strict)
//   SRC 0: A tiles are plain bf16 dz. SRC 1: A tiles are the bf16 logits store; transform warps 1-3 write bf16 dz
//   (ce_dz: the same bits as every other stage) into one of NDZ shared tiles of the same layout, with the row
//   statistics (lse, g, target) loaded by TMA next to each stage. B1.5 (dgrad, own_rows): a CTA runs every N tile of
//   its row block and the transform also writes the dz chunk over the store in the last N pass. Each dz element is
//   computed once per CTA tile, so with N = 512 the transform costs ~0.44 of the tensor time (the reason for the
//   2 x 256 tile).
// Warpgroup 0: warp 0 lane 0 the TMA producer, warps 1-3 the transform; warpgroups 1, 2: consumers on the same tile
// (consumer c: columns [c WN, (c + 1) WN) of the tile). No split-K: each output element is one sequential K loop, so dX
// of a row does not depend on the row count and both outputs are bitwise repeatable.
constexpr int G9_BM = 64, G9_BK = 32, G9_S = 5, G9_NDZ = 4;
constexpr int G9_META = 768;   // per stage: lse, g, target slots of 256 B (TMA destinations 128-aligned)

struct Gemm9 {
  CUtensorMap tm_a, tm_b, tm_lse, tm_gs, tm_tg;
  int rows, D, V;
  int n_mb, n_nb, KB;
  int own_rows;                                     // a CTA runs every N tile of its row block (B1.5's dz write)
  const bf16* dx_old; bf16* dx_out; int levanter;   // dgrad
  bf16* dz_out; int64_t ldz;                        // dgrad, own_rows: bf16 dz written over the store, last N pass
  float* dw; int round_chunk;                       // wgrad
};

// The i-th tile of this CTA. Default order: N tile outer, row block inner over the persistent grid (the CTAs in flight
// read the same B tiles at about the same time). own_rows: work unit = row block, its N tiles in order inside the CTA,
// so the last pass may overwrite the store rows that no other CTA reads.
__device__ __forceinline__ bool g9_tile(const Gemm9& p, int i, int& mb, int& nb) {
  if (p.own_rows) {
    mb = blockIdx.x + (i / p.n_nb) * gridDim.x;
    nb = i % p.n_nb;
    return mb < p.n_mb;
  }
  const int t = blockIdx.x + i * gridDim.x;
  nb = t / p.n_mb;
  mb = t % p.n_mb;
  return t < p.n_mb * p.n_nb;
}

template <int KIND, int SRC, int WN>
struct G9 {
  static constexpr int NB = 2 * WN / 64;     // 64-column B boxes per stage
  static constexpr int kA = G9_BM * G9_BK * 2;
  static constexpr int kBox = G9_BK * 128;
  static constexpr int kB = NB * kBox;
  static constexpr int kMetaRows = KIND == 0 ? G9_BM : G9_BK;
  static constexpr int kTx = kA + kB + (SRC ? 3 * kMetaRows * 4 : 0);
  // the statistics region is padded to 1024 B: the dz tiles after it must sit on the 1024-B swizzle repeat (the
  // hardware swizzles by absolute shared address bits)
  static constexpr int kMetaRegion = SRC ? (G9_S * G9_META + 1023) / 1024 * 1024 : 0;
  static constexpr int kSmem = 1024 + G9_S * (kA + kB) + kMetaRegion + (SRC ? G9_NDZ * kA : 0) +
                               (2 * G9_S + 2 * G9_NDZ) * 8;
};

template <int KIND, int SRC, int WN>
__global__ void __launch_bounds__(384, 1) k_ce_gemm_sm90(const __grid_constant__ Gemm9 p) {
  using C = G9<KIND, SRC, WN>;
  extern __shared__ uint8_t smraw[];
  uint8_t* sm = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
  const uint32_t s0 = smem_u32(sm);
  const uint32_t a_base = s0, b_base = a_base + G9_S * C::kA;
  const uint32_t m_base = b_base + G9_S * C::kB;
  const uint32_t z_base = m_base + C::kMetaRegion;
  const uint32_t bar = z_base + (SRC ? G9_NDZ * C::kA : 0);
  auto full = [&](uint32_t i) { return bar + 8 * i; };
  auto empty = [&](uint32_t i) { return bar + 8 * (G9_S + i); };
  auto dzf = [&](uint32_t i) { return bar + 8 * (2 * G9_S + i); };
  auto dze = [&](uint32_t i) { return bar + 8 * (2 * G9_S + G9_NDZ + i); };
  const int wg = threadIdx.x >> 7, warp_all = threadIdx.x >> 5, lane = threadIdx.x & 31;

  if (threadIdx.x == 0) {
    for (int i = 0; i < G9_S; ++i) {
      mbar_init(full(i), 1);
      mbar_init(empty(i), SRC ? 8 + 3 : 8);   // consumer warps (+ transform warps)
    }
    for (int i = 0; i < G9_NDZ; ++i) {
      mbar_init(dzf(i), 3);
      mbar_init(dze(i), 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  // registers: the CTA starts with 3 x 128 x 168; setmaxnreg.inc only takes what a .dec released, so the split must
  // sum to the same total (88 + 2 x 208 = 3 x 168), else the consumers wait forever
  if (wg == 0) {
    setmaxnreg_dec<88>();
    if (threadIdx.x == 0) {   // ---- producer
      uint32_t it = 0;
      int mb, nb;
      for (int i = 0; g9_tile(p, i, mb, nb); ++i) {
        for (int kb = 0; kb < p.KB; ++kb, ++it) {
          const uint32_t s = it % G9_S, ph = (it / G9_S) & 1;
          mbar_wait(empty(s), ph ^ 1);
          mbar_expect_tx(full(s), C::kTx);
          const int ka = kb * G9_BK;
          if (KIND == 0) tma_load_2d(a_base + s * C::kA, &p.tm_a, ka, mb * G9_BM, full(s));
          else tma_load_2d(a_base + s * C::kA, &p.tm_a, mb * G9_BM, ka, full(s));
#pragma unroll
          for (int b = 0; b < C::NB; ++b)
            tma_load_2d(b_base + s * C::kB + b * C::kBox, &p.tm_b, nb * 2 * WN + b * 64, ka, full(s));
          if (SRC) {
            const int r0 = KIND == 0 ? mb * G9_BM : ka;
            const uint32_t m = m_base + s * G9_META;
            tma_load_1d(m, &p.tm_lse, r0, full(s));
            tma_load_1d(m + 256, &p.tm_gs, r0, full(s));
            tma_load_1d(m + 512, &p.tm_tg, r0, full(s));
          }
        }
      }
    } else if (SRC && warp_all >= 1) {   // ---- transform: store tile -> bf16 dz tile
      const int tt = threadIdx.x - 32;     // 0..95
      uint32_t it = 0;
      int mb, nb;
      for (int i = 0; g9_tile(p, i, mb, nb); ++i) {
        const bool write_dz = KIND == 0 && p.dz_out && nb == p.n_nb - 1;
        for (int kb = 0; kb < p.KB; ++kb, ++it) {
          const uint32_t s = it % G9_S, ph = (it / G9_S) & 1;
          const uint32_t b = it % G9_NDZ, bph = (it / G9_NDZ) & 1;
          mbar_wait(full(s), ph);
          mbar_wait(dze(b), bph ^ 1);
          // every chunk's shared loads first, then all 8 x NQ exponentials as independent chains (one transform warp
          // per scheduler: the latency must be hidden inside the thread), shared memory through 32-bit addresses
          const uint32_t A = a_base + s * C::kA, Z = z_base + b * C::kA, M = m_base + s * G9_META;
          constexpr int NQ = (G9_BM * G9_BK / 8 + 95) / 96;
          uint4 raw[NQ];
          uint32_t off[NQ];
          int rr[NQ], v0s[NQ];
#pragma unroll
          for (int j = 0; j < NQ; ++j) {
            const int q = tt + 96 * j;
            if (KIND == 0) {   // [64 rows][4 chunks], 64B swizzle
              rr[j] = q >> 2;
              const int c = q & 3;
              off[j] = rr[j] * 64 + ((c ^ ((rr[j] >> 1) & 3)) << 4);
              v0s[j] = kb * G9_BK + c * 8;
            } else {           // [32 rows][8 chunks], 128B swizzle
              rr[j] = q >> 3;
              const int c = q & 7;
              off[j] = rr[j] * 128 + ((c ^ (rr[j] & 7)) << 4);
              v0s[j] = mb * G9_BM + c * 8;
            }
            if (q < G9_BM * G9_BK / 8) raw[j] = ld_shared_v4(A + off[j]);
          }
#pragma unroll
          for (int j = 0; j < NQ; ++j) {
            if (tt + 96 * j >= G9_BM * G9_BK / 8) break;
            const int r = rr[j], v0 = v0s[j];
            const float l = ld_shared_f32(M + 4 * r), gs = ld_shared_f32(M + 256 + 4 * r);
            const int tg = int(ld_shared_u32(M + 512 + 4 * r));
            uint32_t* u = reinterpret_cast<uint32_t*>(&raw[j]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int v = v0 + 2 * e;
              const float d0 = ce_dz_sel(bf16lo(u[e]), l, gs, v == tg, v < p.V);
              const float d1 = ce_dz_sel(bf16hi(u[e]), l, gs, v + 1 == tg, v + 1 < p.V);
              u[e] = pack_bf16x2(d0, d1);
            }
            st_shared_v4(Z + off[j], raw[j]);
            if (write_dz) {   // B1.5: dz over the store rows this CTA owns (their last reader)
              const int row = mb * G9_BM + r;
              if (row < p.rows && v0 < p.V) *reinterpret_cast<uint4*>(p.dz_out + int64_t(row) * p.ldz + v0) = raw[j];
            }
          }
          fence_proxy_async();
          __syncwarp();
          if (lane == 0) {
            mbar_arrive(dzf(b));
            mbar_arrive(empty(s));
          }
        }
      }
    }
    return;
  }

  // ---- consumers
  setmaxnreg_inc<208>();
  const int c = wg - 1;
  const int tw = threadIdx.x & 127, warp = tw >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const uint32_t a_tiles = SRC ? z_base : a_base;
  const uint64_t desc_a0 = KIND == 0 ? desc_k64(a_tiles) : desc_mn128(a_tiles, 128 * G9_BK);
  const uint64_t desc_b0 = desc_mn128(b_base + c * (WN / 64) * C::kBox, C::kBox);
  const uint32_t a_lo = desc_lo(desc_a0), a_hi = desc_hi(desc_a0), b_lo = desc_lo(desc_b0), b_hi = desc_hi(desc_b0);
  uint32_t it = 0;
  int mb, nb;
  for (int i = 0; g9_tile(p, i, mb, nb); ++i) {
    float acc[WN / 2];
#pragma unroll
    for (int i = 0; i < WN / 2; ++i) acc[i] = 0.f;
    for (int kb = 0; kb < p.KB; ++kb, ++it) {
      const uint32_t s = it % G9_S, ph = (it / G9_S) & 1;
      mbar_wait(full(s), ph);
      uint32_t ai = s;   // A tile index: the stage, or the dz tile
      if (SRC) {
        ai = it % G9_NDZ;
        mbar_wait(dzf(ai), (it / G9_NDZ) & 1);
      }
      // descriptors = base + (byte offset >> 4), bases built once (see the logits kernel)
      const uint32_t da = a_lo + ((ai * C::kA) >> 4), db = b_lo + ((s * C::kB) >> 4);
      wgmma_fence();
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint64_t dbk = desc_join(db + (2048 >> 4) * ks, b_hi);
        const uint64_t dak = desc_join(da + (KIND == 0 ? (32 >> 4) : (2048 >> 4)) * ks, a_hi);
        if constexpr (WN == 256) wgmma_m64n256<KIND, 1>(acc, dak, dbk, 1);
        else wgmma_m64n192<KIND, 1>(acc, dak, dbk, 1);
      }
      wgmma_commit();
      wgmma_wait<1>();
      if (kb > 0 && lane == 0) {
        mbar_arrive(empty((it - 1) % G9_S));
        if (SRC) mbar_arrive(dze((it - 1) % G9_NDZ));
      }
    }
    wgmma_wait<0>();
    if (lane == 0) {
      mbar_arrive(empty((it - 1) % G9_S));
      if (SRC) mbar_arrive(dze((it - 1) % G9_NDZ));
    }
#pragma unroll
    for (int i = 0; i < WN / 2; ++i) fence_regs(acc[i]);

    // ---- epilogue
    const int col0 = nb * 2 * WN + c * WN;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = mb * G9_BM + warp * 16 + g + 8 * r;
      if (KIND == 0 ? row >= p.rows : row >= p.V) continue;
#pragma unroll
      for (int j = 0; j < WN / 8; ++j) {
        const int col = col0 + j * 8 + 2 * t4;
        if (col >= p.D) continue;
        float a0 = acc[j * 4 + r * 2], a1 = acc[j * 4 + r * 2 + 1];
        if (KIND == 0) {
          const int64_t o = int64_t(row) * p.D + col;
          if (p.dx_old) {
            const uint32_t ov = *reinterpret_cast<const uint32_t*>(p.dx_old + o);
            if (p.levanter) {
              a0 = __bfloat162float(__float2bfloat16_rn(a0));
              a1 = __bfloat162float(__float2bfloat16_rn(a1));
            }
            a0 = __fadd_rn(bf16lo(ov), a0);
            a1 = __fadd_rn(bf16hi(ov), a1);
          }
          *reinterpret_cast<uint32_t*>(p.dx_out + o) = pack_bf16x2(a0, a1);
        } else {
          if (p.round_chunk) {
            a0 = __bfloat162float(__float2bfloat16_rn(a0));
            a1 = __bfloat162float(__float2bfloat16_rn(a1));
          }
          float2* o = reinterpret_cast<float2*>(p.dw + int64_t(row) * p.D + col);
          float2 cur = *o;
          cur.x = __fadd_rn(cur.x, a0);
          cur.y = __fadd_rn(cur.y, a1);
          *o = cur;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ host side
PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  static std::once_flag once;
  std::call_once(once, [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult q;
    NM_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &f, cudaEnableDefault, &q));
    NM_REQUIRE(f && q == cudaDriverEntryPointSuccess, "fused CE: cuTensorMapEncodeTiled is not available");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(f);
  });
  return fn;
}

// Tensor map of a row-major 2-D array (inner dimension `inner` elements contiguous, `outer` rows `ld` elements apart),
// box [box_outer x box_inner], or of a 1-D array (outer == 0). Out-of-bounds elements of a load are zero.
CUtensorMap make_map(CUtensorMapDataType dt, int esize, const void* base, uint64_t inner, uint64_t outer, uint64_t ld,
                     uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swz) {
  CUtensorMap m;
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {ld * esize};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t es[2] = {1, 1};
  const cuuint32_t rank = outer ? 2 : 1;
  const CUresult r = encode_fn()(&m, dt, rank, const_cast<void*>(base), dims, strides, box, es,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  NM_REQUIRE(r == CUDA_SUCCESS, "fused CE: cuTensorMapEncodeTiled failed (" + std::to_string(int(r)) + ")");
  return m;
}
CUtensorMap make_map(const void* base, uint64_t inner, uint64_t outer, uint64_t ld, uint32_t box_inner,
                     uint32_t box_outer) {
  return make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, inner, outer, ld, box_inner, box_outer,
                  CU_TENSOR_MAP_SWIZZLE_128B);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

bool launch_logits_stats_sm90(DType store_dt, const LogitsArgs& a, cudaStream_t st) {
  if (a.bias || (a.store && store_dt != DType::bf16)) return false;   // f32 store, bias: portable kernel
  if (a.D % 64 != 0 || a.D / L9_BK < L9_S || !aligned16(a.x) || !aligned16(a.w)) return false;
  if (a.store && (!aligned16(a.store) || a.ldS % 8 != 0)) return false;
  Logits9 p{};
  p.tm_x = make_map(a.x, a.D, a.rows, a.D, 64, L9_BM);
  p.tm_w = make_map(a.w, a.D, a.V, a.D, 64, L9_BN);
  if (a.store) p.tm_st = make_map(a.store, a.V, a.rows, a.ldS, 64, L9_BM);
  p.targets = a.targets;
  p.tmax = a.tmax;
  p.tsum = a.tsum;
  p.zt = a.zt;
  p.ldT = a.ldT;
  p.rows = a.rows;
  p.V = a.V;
  p.KB = a.D / L9_BK;
  p.n_mb = (a.rows + L9_BM - 1) / L9_BM;
  p.n_nb = (a.V + L9_BN - 1) / L9_BN;
  p.group_m = std::max(1, std::min(p.n_mb, int((16ll << 20) / (int64_t(L9_BM) * a.D * 2))));
  p.write_store = a.store != nullptr;
  p.fast_exp = a.fast_exp;
  // a per-device function attribute (one thread per GPU): set on every launch, legal during graph capture
  NM_CUDA(cudaFuncSetAttribute(k_ce_logits_sm90, cudaFuncAttributeMaxDynamicSharedMemorySize, L9_SMEM));
  k_ce_logits_sm90<<<std::min(p.n_mb * p.n_nb, sm_count()), 384, L9_SMEM, st>>>(p);
  NM_CUDA(cudaGetLastError());
  return true;
}

namespace {

int gemm_wn(int D) { return D % 512 == 0 ? 256 : (D % 384 == 0 ? 192 : 0); }

template <int KIND, int SRC, int WN>
void launch_gemm9(const Gemm9& p, cudaStream_t st) {
  using C = G9<KIND, SRC, WN>;
  auto k = k_ce_gemm_sm90<KIND, SRC, WN>;
  NM_CUDA(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem));   // per device
  const int grid = std::min(p.own_rows ? p.n_mb : p.n_mb * p.n_nb, sm_count());
  k<<<grid, 384, C::kSmem, st>>>(p);
  NM_CUDA(cudaGetLastError());
}

template <int KIND>
void dispatch_gemm9(bool src, int wn, const Gemm9& p, cudaStream_t st) {
  if (src) {
    if (wn == 256) launch_gemm9<KIND, 1, 256>(p, st);
    else launch_gemm9<KIND, 1, 192>(p, st);
  } else {
    if (wn == 256) launch_gemm9<KIND, 0, 256>(p, st);
    else launch_gemm9<KIND, 0, 192>(p, st);
  }
}

// the 1-D TMA maps of the row statistics need 16-byte aligned bases (a chunk offset r0 not a multiple of 4, or an
// unaligned caller pointer, falls back to the portable kernel)
bool meta_aligned(const float* lse, const float* gs, const int32_t* tg) {
  return aligned16(lse) && aligned16(gs) && aligned16(tg);
}

void meta_maps(Gemm9& p, const float* lse, const float* gs, const int32_t* tg, int rows, uint32_t box) {
  p.tm_lse = make_map(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, lse, rows, 0, 0, box, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  p.tm_gs = make_map(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, gs, rows, 0, 0, box, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  p.tm_tg = make_map(CU_TENSOR_MAP_DATA_TYPE_INT32, 4, tg, rows, 0, 0, box, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
}

}  // namespace

bool launch_dgrad_sm90(DType store_dt, const DgradArgs& a, cudaStream_t st) {
  // plain dz or a bf16 store (f32 stores stay on the portable kernel)
  const bool src = store_dt == DType::bf16;
  if (!(src || store_dt == DType::u8)) return false;
  const int wn = gemm_wn(a.D);
  if (!wn || !aligned16(a.store) || !aligned16(a.w) || a.ldS % 8 != 0 || !aligned16(a.dx_out)) return false;
  if (src && !meta_aligned(a.lse, a.gscale, a.targets)) return false;
  Gemm9 p{};
  p.tm_a = make_map(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, a.store, a.V, a.rows, a.ldS, 32, G9_BM,
                    CU_TENSOR_MAP_SWIZZLE_64B);
  p.tm_b = make_map(a.w, a.D, a.V, a.D, 64, G9_BK);
  if (src) meta_maps(p, a.lse, a.gscale, a.targets, a.rows, G9_BM);
  p.rows = a.rows;
  p.D = a.D;
  p.V = a.V;
  p.n_mb = (a.rows + G9_BM - 1) / G9_BM;
  p.n_nb = (a.D + 2 * wn - 1) / (2 * wn);
  p.KB = (a.V + G9_BK - 1) / G9_BK;
  p.dx_old = static_cast<const bf16*>(a.dx_old);
  p.dx_out = static_cast<bf16*>(a.dx_out);
  p.levanter = a.r == Rounding::Levanter;
  if (a.write_dz) {
    p.own_rows = 1;
    p.dz_out = static_cast<bf16*>(const_cast<void*>(a.store));
    p.ldz = a.ldS;
  }
  dispatch_gemm9<0>(src, wn, p, st);
  return true;
}

bool launch_wgrad_sm90(DType store_dt, const WgradArgs& a, cudaStream_t st) {
  const bool src = store_dt == DType::bf16;
  if (!(src || store_dt == DType::u8)) return false;
  const int wn = gemm_wn(a.D);
  if (!wn || !aligned16(a.store) || !aligned16(a.x) || a.ldS % 8 != 0) return false;
  if (src && !meta_aligned(a.lse, a.gscale, a.targets)) return false;
  Gemm9 p{};
  p.tm_a = make_map(a.store, a.V, a.rows, a.ldS, 64, G9_BK);
  p.tm_b = make_map(a.x, a.D, a.rows, a.D, 64, G9_BK);
  if (src) meta_maps(p, a.lse, a.gscale, a.targets, a.rows, G9_BK);
  p.rows = a.rows;
  p.D = a.D;
  p.V = a.V;
  p.n_mb = (a.V + G9_BM - 1) / G9_BM;
  p.n_nb = (a.D + 2 * wn - 1) / (2 * wn);
  p.KB = (a.rows + G9_BK - 1) / G9_BK;
  p.dw = a.dw;
  p.round_chunk = a.epi_round_chunk;
  dispatch_gemm9<1>(src, wn, p, st);
  return true;
}

}  // namespace nm::ce
