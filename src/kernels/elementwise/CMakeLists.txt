# src/kernels/elementwise: stage S3 (unfused elementwise kernels, bf16 and f32 storage forms, C ABI); DESIGN.md §3.0-§3.6.
# All kernels are portable (sm_90a + sm_80). Contraction (DESIGN.md §3.0): the units that instantiate the levanter-mode
# kernels build with --fmad=false (levanter rounding points, explicit contractions); the accurate-mode instantiations
# (*_acc.cu) keep nvcc's default contraction, as does rope_tables.cu, so its device method evaluates libdevice powf /
# cosf / sinf as XLA:GPU does.
nm_module(kernels/elementwise
  PORTABLE rmsnorm.cu rmsnorm_acc.cu qknorm_rope.cu qknorm_rope_acc.cu swiglu.cu swiglu_acc.cu embed.cu doc_meta.cu
           reduce_cast.cu rope_tables.cu capi_elementwise.cpp
  FMAD_OFF rmsnorm.cu qknorm_rope.cu swiglu.cu embed.cu doc_meta.cu reduce_cast.cu)
