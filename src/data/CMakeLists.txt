# src/data: stage S1 (converted datasets, threefry, LevanterOrder, DataPlan, prefetch ring + pinned slots); DESIGN.md §5.
nm_module(data
  PORTABLE json.cpp order.cpp dataset.cpp plan.cpp batch.cpp loader.cpp capi_data.cpp threefry_device.cu)
