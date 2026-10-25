/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include<cute/arch/copy_sm90_tma.hpp>

namespace cute
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_BULK_REDUCE_ADD
{
  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr,
       float      * gmem_ptr, int32_t store_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr,
       float      * gmem_ptr, int32_t store_bytes, uint64_t cache_hint)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [%0], [%1], %2, %3;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// marincpp: plain bulk store (the first head of a GQA group in deterministic mode stores instead of adding)
struct SM90_BULK_STORE
{
  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr,
       float      * gmem_ptr, int32_t store_bytes, uint64_t cache_hint)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_STORE without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// marincpp: deterministic hand-off of a global tile between bulk writers ordered by a semaphore. tma_store_wait
// (cp.async.bulk.wait_group.read) returns once shared memory has been read, not when the global add has landed; the
// writer must wait for its writes (wait_group 0) and order them before its generic-proxy release (proxy fence), and
// the next writer orders its acquire before its own async-proxy access.
struct SM90_BULK_HANDOFF
{
  CUTE_HOST_DEVICE static void
  release_ready()
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile("cp.async.bulk.wait_group 0;\n"
                 "fence.proxy.async.global;\n" : : : "memory");
#endif
  }

  CUTE_HOST_DEVICE static void
  after_acquire()
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    asm volatile("fence.proxy.async.global;\n" : : : "memory");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
