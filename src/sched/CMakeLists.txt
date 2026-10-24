# src/sched: stage S2, the schedule tables (DESIGN.md §4.4): fixture loader of record, host formula (a port of the
# XLA:GPU arithmetic of the levanter schedules) as cross-check and non-parity fallback, C ABI nm_schedule_tables.
nm_module(sched PORTABLE schedule.cpp xla_math.cpp capi_sched.cpp)
