# src/kernels/attention: stage S4 (FA3 entry points and backward schedulers, reference attention); DESIGN.md §3.4.
#   PORTABLE  attention.cpp (backends), capi_attention.cpp (C ABI), ref_attention.cu (RefF32 / test oracle, sm_80 too)
#   SM90A     fa3_backend.cu (parameter fill, work lists, doc tables) and one unit per FA3 kernel variant
#             (fa3_inst_*.cu, see fa3_internal.hpp), plus FA3's flash_prepare_scheduler.cu (varlen forward metadata),
#             compiled from third_party/fa3-hopper with FA3's own flags
#             (hopper/setup.py): --use_fast_math stays confined to these units (DESIGN.md §13.1).
file(GLOB _fa3_units CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fa3_inst_*.cu)
nm_module(kernels/attention
  PORTABLE attention.cpp capi_attention.cpp ref_attention.cu
  SM90A fa3_backend.cu ${_fa3_units} ${PROJECT_SOURCE_DIR}/third_party/fa3-hopper/flash_prepare_scheduler.cu
  DEPS CUDA::cudart)
set(NM_FA3_INCLUDE ${PROJECT_SOURCE_DIR}/third_party/fa3-hopper ${PROJECT_SOURCE_DIR}/third_party/cutlass/include)
target_include_directories(nm_kernels_attention_sm90a PRIVATE ${NM_FA3_INCLUDE})
target_compile_definitions(nm_kernels_attention_sm90a PRIVATE CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED
                           CUTLASS_ENABLE_GDC_FOR_SM90 CUTLASS_DEBUG_TRACE_LEVEL=0 NDEBUG)
target_compile_options(nm_kernels_attention_sm90a PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_source_files_properties(${_fa3_units} PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
