# Compile flags of every marincpp translation unit (DESIGN.md §3.0 FMA policy, §13).
#   host code (C, C++ and the host side of .cu): -Wall -Werror, -ffp-contract=off (no FMA contraction on the host)
#   device code: nvcc default contraction; a module lists its levanter-mode elementwise / optimizer / init / eval
#   units under FMAD_OFF (nm_module) to build them with --fmad=false and explicit __fmaf_rn where XLA contracts.
add_library(nm_flags INTERFACE)
target_compile_options(nm_flags INTERFACE
  $<$<COMPILE_LANGUAGE:C>:-Wall -Werror -ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror -ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-ffp-contract=off --Werror=all-warnings>)
# __FILE__ in error messages is repo-relative (src/config/config.cpp:264): no build path in binaries or run logs.
target_compile_options(nm_flags INTERFACE
  $<$<COMPILE_LANGUAGE:C,CXX>:-fmacro-prefix-map=${PROJECT_SOURCE_DIR}/=>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fmacro-prefix-map=${PROJECT_SOURCE_DIR}/=>)
target_include_directories(nm_flags INTERFACE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
