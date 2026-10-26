# src/model: stage S7 (the baseline model: parameter table, buffers, forward + hand-written backward); DESIGN.md §3.
# npy.cpp reads the levanter fixtures for the S7 gates (test-side only; the training binary reads no .npy).
nm_module(model
  PORTABLE baseline.cpp fixture.cpp npy.cpp
  DEPS CUDA::cublasLt)
