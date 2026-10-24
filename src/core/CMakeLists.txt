# src/core: errors, hashing, registries, device probes, C ABI basics (stage S0).
nm_module(core
  PORTABLE core.cpp sha256.cpp registry.cpp capi_core.cpp device_probe.cu
  SM90A device_probe_sm90a.cu)
# capi_core.cpp includes the build-time git sha (cmake/GitVersion.cmake).
target_include_directories(nm_core PRIVATE ${NM_GENERATED_DIR})
add_dependencies(nm_core nm_git_sha)
