#pragma once
#include <cstdio>
#include <cstdlib>
#include "../include/CudaUtils.hpp"

cutensorAlgo_t chooseContractionAlgo()
{
// TODO: CUTENSOR_ALGO_DEFAULT runs internal heurisitics to pick a good algorithm, but I want to understand it
// to have finer control over it
	cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
	return algo;
}
	
cutensorPlan_t makeContractionPlan(cutensorOperationDescriptor_t descOp, cutensorHandle_t handle)
{
	cutensorPlan_t plan;
	cutensorPlanPreference_t planPref;					
	// plan holds the work to be done, in this case a contraction, along with all the parameters we tune now
	// planPref narrows the choices of algorithms/variants/kernels to use
  	
	cutensorAlgo_t algo = chooseContractionAlgo();				// default is CUTENSOR_ALGO_DEFAULT
	
	HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(handle, 
				&planPref, 					// mode and algo go into planPref
				algo,
				CUTENSOR_JIT_MODE_NONE));			// Toggle JIT compilation
//			        CUTENSOR_JIT_MODE_NONE);			

	uint64_t workspaceSizeEstimate{0};					// outputs estimated size to this
  	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
	HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(handle,
			       descOp,
			       planPref,
			       workspacePref,
			       &workspaceSizeEstimate));			

	HANDLE_CUTENSOR_ERROR(cutensorCreatePlan(handle,
		    &plan,
		    descOp,
		    planPref,
		    workspaceSizeEstimate));
	return plan;
}
