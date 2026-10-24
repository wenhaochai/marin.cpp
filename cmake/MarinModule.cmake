# nm_module(<name>
#           [PORTABLE <src>...]   built for NM_ARCHS_PORTABLE (sm_90a + sm_80); host .cpp files go here too
#           [SM90A <src>...]      built for NM_ARCHS_SM90A only (WGMMA / TMA / FA3); never run on the vis node
#           [FMAD_OFF <src>...]   subset of the above compiled with --fmad=false (levanter-mode units, §3.0)
#           [DEPS <target>...])   usage requirements (CUDA::cublasLt, nm::nccl, ...)
#
# Creates OBJECT libraries nm_<name> (portable) and nm_<name>_sm90a (if SM90A sources exist), with <name>'s slashes
# replaced by underscores, and appends them to the global property NM_MODULE_TARGETS that the final targets link. A
# module without sources (a stub) creates nothing.
function(nm_module name)
  cmake_parse_arguments(A "" "" "PORTABLE;SM90A;FMAD_OFF;DEPS" ${ARGN})
  string(REPLACE "/" "_" id "${name}")
  foreach(f IN LISTS A_FMAD_OFF)
    set_property(SOURCE ${f} APPEND PROPERTY COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
  endforeach()
  if(A_PORTABLE)
    add_library(nm_${id} OBJECT ${A_PORTABLE})
    set_target_properties(nm_${id} PROPERTIES CUDA_ARCHITECTURES "${NM_ARCHS_PORTABLE}")
    target_link_libraries(nm_${id} PUBLIC nm_flags CUDA::cudart ${A_DEPS})
    set_property(GLOBAL APPEND PROPERTY NM_MODULE_TARGETS nm_${id})
  endif()
  if(A_SM90A)
    add_library(nm_${id}_sm90a OBJECT ${A_SM90A})
    set_target_properties(nm_${id}_sm90a PROPERTIES CUDA_ARCHITECTURES "${NM_ARCHS_SM90A}")
    target_link_libraries(nm_${id}_sm90a PUBLIC nm_flags CUDA::cudart ${A_DEPS})
    set_property(GLOBAL APPEND PROPERTY NM_MODULE_TARGETS nm_${id}_sm90a)
  endif()
endfunction()
