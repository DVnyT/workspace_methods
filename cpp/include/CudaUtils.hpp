#pragma once

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"

// TODO: Some CUDA wrappers for memory safety!

extern cutensorHandle_t globalHandle;

cutensorAlgo_t chooseContractionAlgo();
cutensorPlan_t makeContractionPlan(cutensorOperationDescriptor_t descOp);

