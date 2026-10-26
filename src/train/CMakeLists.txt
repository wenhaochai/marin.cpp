# src/train: stage S7 (optimizer binding of the baseline model, the multi-GPU step loop with one graph per rank);
# DESIGN.md §1, §2.4, §13.3. The trainer needs NCCL (top-level NM_HAVE_NCCL).
if(NM_HAVE_NCCL)
  nm_module(train PORTABLE optimizer.cpp trainer.cpp DEPS CUDA::cublasLt nm::nccl)
else()
  nm_module(train PORTABLE optimizer.cpp DEPS CUDA::cublasLt)
endif()
