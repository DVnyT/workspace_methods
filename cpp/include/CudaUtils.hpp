#pragma once

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverDn.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cublas_v2.h"

// TODO: Some CUDA wrappers for memory safety!

#define HANDLE_ERROR(x)                                               	\
{ const auto err = x;                                                 	\
  if( err != CUTENSOR_STATUS_SUCCESS )                                	\
  { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } 	\
};

#define HANDLE_CUDA_ERROR(x)                                      	\
{ const auto err = x;                                             	\
  if( err != cudaSuccess )                                        	\
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } 	\
};

extern cutensorHandle_t globalHandle;

struct CusolverHandle
{

};


struct CutensorHandle
{

};

struct CublasHandle
{

};

struct StreamPool
{

};

cutensorAlgo_t chooseContractionAlgo();
cutensorPlan_t makeContractionPlan(cutensorOperationDescriptor_t descOp);

