#pragma once

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cublas_v2.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverDn.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"
#include <cstdlib>

// TODO: Some CUDA wrappers for memory safety!

#define HANDLE_CUTENSOR_ERROR(x)                                                                                 \
        {                                                                                                        \
                const auto err = x;                                                                              \
                if (err != CUTENSOR_STATUS_SUCCESS)                                                              \
                {                                                                                                \
                        printf("cuTENSOR error %s at %s:%d\n", cutensorGetErrorString(err), __FILE__, __LINE__); \
                        exit(-1);                                                                                \
                }                                                                                                \
        };

#define HANDLE_CUDA_ERROR(x)                                                                             \
        {                                                                                                \
                const auto err = x;                                                                      \
                if (err != cudaSuccess)                                                                  \
                {                                                                                        \
                        printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
                        exit(-1);                                                                        \
                }                                                                                        \
        };

#define HANDLE_CUSOLVER_ERROR(x)                                                         \
        {                                                                                \
                const auto err = x;                                                      \
                if (err != cudaSuccess)                                                  \
                {                                                                        \
                        printf("cuSOLVER error %d at %s:%d\n", err, __FILE__, __LINE__); \
                        exit(-1);                                                        \
                }                                                                        \
        };

#define HANDLE_CUBLAS_ERROR(x)                                                         \
        {                                                                              \
                const auto err = x;                                                    \
                if (err != cudaSuccess)                                                \
                {                                                                      \
                        printf("cuBLAS error %d at %s:%d\n", err, __FILE__, __LINE__); \
                        exit(-1);                                                      \
                }                                                                      \
        };

#define HANDLE_CUTENSORNET_ERROR(x)                                                                               \
        {                                                                                                         \
                const auto err = x;                                                                               \
                if (err != cudaSuccess)                                                                           \
                {                                                                                                 \
                        printf("cuBLAS error %d at %s:%d\n", cutensornetGetErrorString(err), __FILE__, __LINE__); \
                        exit(-1);                                                                                 \
                }                                                                                                 \
        };

cutensorAlgo_t chooseContractionAlgo();
cutensorPlan_t makeContractionPlan(cutensorOperationDescriptor_t descOp, cutensorHandle_t handle);
