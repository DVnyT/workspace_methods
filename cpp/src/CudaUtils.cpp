#pragma once
#include <cstdio>
#include <cstdlib>
#include "../include/CudaUtils.hpp"

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};
cutensorAlgo_t chooseContractionAlgo()
{
// TODO: CUTENSOR_ALGO_DEFAULT runs internal heurisitics to pick a good algorithm, but I want to understand it
// to have finer control over it
	cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
	return algo;
}
	
cutensorPlan_t makeContractionPlan(cutensorOperationDescriptor_t descOp)
{
	cutensorPlan_t plan;
	cutensorPlanPreference_t planPref;					
	// plan holds the work to be done, in this case a contraction, along with all the parameters we tune now
	// planPref narrows the choices of algorithms/variants/kernels to use
  	
	cutensorAlgo_t algo = chooseContractionAlgo();				// default is CUTENSOR_ALGO_DEFAULT
	
	HANDLE_ERROR(cutensorCreatePlanPreference(globalHandle, 
				&planPref, 					// mode and algo go into planPref
				algo,
				CUTENSOR_JIT_MODE_NONE));			// Toggle JIT compilation
//			        CUTENSOR_JIT_MODE_DEFAULT);			

	uint64_t workspaceSizeEstimate{0};					// outputs estimated size to this
  	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
	HANDLE_ERROR(cutensorEstimateWorkspaceSize(globalHandle,
			       descOp,
			       planPref,
			       workspacePref,
			       &workspaceSizeEstimate));			

	HANDLE_ERROR(cutensorCreatePlan(globalHandle,
		    &plan,
		    descOp,
		    planPref,
		    workspaceSizeEstimate));
	return plan;
}
