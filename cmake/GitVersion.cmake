# NM_GIT_SHA: short commit of the source tree, "+dirty" if tracked files changed or untracked files exist under src/,
# include/, cmake/ or CMakeLists.txt. Generated at BUILD time (target nm_git_sha runs cmake/GitSha.cmake on every build
# and rewrites ${NM_GENERATED_DIR}/nm_git_sha.h only when the text changes), so nm_version() follows commits made
# without a reconfigure. nm_version() returns it; run manifests record it.
set(NM_GENERATED_DIR ${PROJECT_BINARY_DIR}/generated)
set(_nm_git_sha_cmd ${CMAKE_COMMAND} -DNM_SRC=${PROJECT_SOURCE_DIR} -DNM_OUT=${NM_GENERATED_DIR}/nm_git_sha.h
    -P ${PROJECT_SOURCE_DIR}/cmake/GitSha.cmake)
execute_process(COMMAND ${_nm_git_sha_cmd})   # the header exists before the first build
add_custom_target(nm_git_sha ALL COMMAND ${_nm_git_sha_cmd} BYPRODUCTS ${NM_GENERATED_DIR}/nm_git_sha.h VERBATIM)
