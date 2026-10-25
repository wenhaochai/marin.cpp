/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>

#include <stdexcept>
#include <string>

#include <cutlass/cutlass.h>

#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            /* marincpp: throw (the caller reports it) instead of exit(1) */                                \
            throw std::runtime_error(std::string("FA3 CUDA error (" __FILE__ ":") + std::to_string(__LINE__) \
                                     + "): " + cudaGetErrorString(status_));                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())

#define CHECK_CUTLASS(call)                                                                               \
    do {                                                                                                  \
        cutlass::Status status_ = (call);                                                                 \
        if (status_ != cutlass::Status::kSuccess) {                                                        \
            /* marincpp: throw (the caller reports it) instead of exit(1) */                                \
            throw std::runtime_error(std::string("FA3 CUTLASS error (" __FILE__ ":") + std::to_string(__LINE__) \
                                     + "): " + cutlass::cutlassGetStatusString(status_));                 \
        }                                                                                                 \
    } while(0)
