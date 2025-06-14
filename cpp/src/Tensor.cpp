#pragma once

#include "../include/Tensor.hpp"
#include <cstdint>

Tensor::Tensor() = default; 

// DONE:  TODO: Move the logic outside the .hpp 

Tensor::Tensor(const std::map<uintptr_t, int64_t>& lookup)	// lookup = {(key,value): (Index address, Index dim)}
{
	m_order = lookup.size();				// i.e. number of Indices
	
	for (const auto& i: lookup)
	{
		m_modes.push_back(i.first);
		m_extents.push_back(i.second); 
		m_elements *= i.second;
	}
		
	m_byteSize = m_elements * sizeof(float);
	
	if (m_byteSize != 0) 
	{
		m_pHost = (float*) malloc(m_byteSize);
		cudaMalloc((void**)& m_pDevice, m_byteSize);
		
		for(int j = 0; j < m_elements; j++)		// populate the tensor
		{
			m_pHost[j] = ((float) rand())/RAND_MAX + 0.5;
		}

		cudaMemcpy(m_pDevice, m_pHost, m_byteSize, cudaMemcpyHostToDevice);		// copy tensor to GPU
	}
}
	
Tensor::Tensor(const std::vector<uintptr_t>& modes, const std::vector<int64_t>& extents)	// alternate const.
: m_modes(modes), m_extents(extents)
{
	m_order = modes.size();
	for (const auto& i : extents)
	{
		m_elements *= i;
	}
		
	m_size = m_elements * sizeof(float);
	if (m_size != 0) 
	{
		m_pHost = (float*) malloc(m_byteSize);
		cudaMalloc((void**)& m_pDevice, m_byteSize);
		
		
		for(int j = 0; j < m_elements; j++)
		{
			m_pHost[j] = ((float) rand())/RAND_MAX + 0.5;			// populate the tensor
		}

		cudaMemcpy(m_pDevice, m_pHost, m_byteSize, cudaMemcpyHostToDevice);
	}

}

Tensor::Tensor Contract(const Tensor::Tensor& A, const Tensor::Tensor& B, cutensorHandle_t& globalHandle)
{
	cutensorTensorDescriptor_t descA;				// Allocated tensor descriptor
	cutensorTensorDescriptor_t descB;
	
	const uint32_t kAlignment = 128;  				// TODO: Do make this a global variable!
	
	const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;	// Precision of contraction
	
	cutensorCreateTensorDescriptor_t(globalHandle,			// Initializes cuTENSOR's library context
				  &descA,				
				  A.m_order,
				  A.m_extents.data(),		
				  NULL,					// Stride (refer below!)		
				  CUTENSOR_R_32F,		
				  kAlignment);
	
	/*
		A comment about Stride:

	*/

	cutensorCreateTensorDescriptor_t(globalHandle,
				  &descB,
				  B.m_order,
				  B.m_extents.data(),
				  NULL,
				  CUTENSOR_R_32F,
				  kAlignment);
	
	cutensorCreateTensorDescriptor_t(globalHandle,			// Output Tensor C for a simple contraction
				  &descC
				  C.m_order,
				  C.m_extents.data(),
				  NULL,
				  CUTENSOR_R_32F,
				  kAlignment);			

	/*
		The Operation Descriptor below encodes the contraction, 
			D_modesD = alpha * opA(A_modesA) * opB(B_modesB) + beta * opC(C_modesC)
		For our two-tensor contraction, we set beta to 0, and reuse our Tensor Descriptor for C, as the
		output Tensor D.
	*/
	
	
  	cutensorOperationDescriptor_t descOp;
	
	cutensorCreateContraction(globalHandle,
			   &descOp,
			   descA, A.m_modes.data(), CUTENSOR_OP_IDENTITY,	// descA, A.m_modes, opA
			   descB, B.m_modes.data(), CUTENSOR_OP_IDENTITY,
			   descC, C.m_modes.data(), CUTENSOR_OP_IDENTITY,
			   descC, C.m_modes.data(),
			   descCompute);

	
  	typedef float floatTypeCompute;
  	floatTypeCompute alpha = (floatTypeCompute)1.0f;
  	floatTypeCompute beta = (floatTypeCompute)0.f;

  	const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
	cutensorPlanPreference_t planPref;
  	
	cutensorCreatePlanPreference(handle, 
				&planPref, 
				algo,
				CUTENSOR_JIT_MODE_NONE));

	uint64_t workspaceSizeEstimate{0};
  	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
	cutensorEstimateWorkspaceSize(globalHandle,
			       &descOp,
			       planPref,
			       workspacePref,
			       &workspaceSizeEstimate);

	cutensorPlan_t plan;
	cutensorCreatePlan(globalHandle,
		    &plan,
		    descOp,
		    planPref,
		    workspaceSizeEstimate);

	cudaStream_t steam;
	cudaStreamCreate(&stream);

  	double minTimeCUTENSOR = 1e100;
  	
	for (int i = 0; i < 3; ++i) 
	{
    		cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
    		cudaDeviceSynchronize();

    		GPUTimer timer;
    		timer.start();

    		cutensorContract(handle, 
		       plan, 
		       (void *)&alpha, A_d, B_d,
		       (void *)&beta, C_d, C_d, 
		       work, actualWorkspaceSize, stream);

    		auto time = timer.seconds();
    		minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
  	}

  

  	double transferedBytes = sizeC + sizeA + sizeB;
  	transferedBytes += ((float)beta != 0.f) ? sizeC : 0;
  	transferedBytes /= 1e9;
	
  	printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", 
	  gflops / minTimeCUTENSOR,
	  transferedBytes / minTimeCUTENSOR);



	cutensorDestroyTensorDescriptor();
	cutensorDestroyTensorDescriptor();
	cutensorDestroyTensorDescriptor();
	
	cutensorDestroyOperationDescriptor();


}


