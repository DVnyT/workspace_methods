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
	/* 
	 *	Step 1: Describe the tensors in a suitable format 
	*/

	cutensorTensorDescriptor_t descA;				// Allocated tensor descriptor for A
	cutensorTensorDescriptor_t descB;

	// TODO: Create a lookup table to initialize indices left in the output Tensor C
	std::map<uintptr_t, int64_t> lookupC;

	Tensor::Tensor C(&lookupC);					// Output Tensor
	
	const uint32_t kAlignment = 128;  				// TODO: Do make this a global variable!
		
	cutensorCreateTensorDescriptor_t(globalHandle,			
				  &descA,				
				  A.m_order,
				  A.m_extents.data(),		
				  NULL,					// Stride (refer below!)		
				  CUTENSOR_R_32F,			// Datatype: 32-bit Real Floats
				  kAlignment);
	
	/*
	*	A comment about Stride:
	*
	*
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
	*	Step 2: Describe the operation to be done on input tensors; create a plan preference, estimate 
	*	workspace size needed over the course of the operation, then create a proper plan that has information
	*	about all tunable parameters
	*	
	*	The Operation Descriptor below encodes the contraction, 
	*		D_modesD = alpha * opA(A_modesA) * opB(B_modesB) + beta * opC(C_modesC)
	*	
	*	For our two-tensor contraction, we set beta to 0, and reuse our Tensor Descriptor for C, as the
	*	output Tensor D. Note that modes_D are equivalent to modes_C.
	*	
	*	opA() does an operation on all elements of A, we currently use the identity operation.
	*/
	
  	cutensorOperationDescriptor_t descOp;			// will encode the operation, init. by function below
	cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;	// Precision of contraction

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

	cutensorPlan_t plan;
	cutensorPlanPreference_t planPref;					
	// plan holds the work to be done, in this case a contraction, along with parameters we tune now
	// planPref narrows the choices of algorithms/variants/kernels to use
  	
	const cutensorAlgo_t algo = chooseContractionAlgo();			// default is CUTENSOR_ALGO_DEFAULT
	
	cutensorCreatePlanPreference(globalHandle, 
				&planPref, 
				algo,
				CUTENSOR_JIT_MODE_NONE));			// Not using Just-In-Time compilation

	uint64_t workspaceSizeEstimate{0};
  	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
	cutensorEstimateWorkspaceSize(globalHandle,
			       &descOp,
			       planPref,
			       workspacePref,
			       &workspaceSizeEstimate);

	cutensorCreatePlan(globalHandle,
		    &plan,
		    descOp,
		    planPref,
		    workspaceSizeEstimate);
	
	uint64_t actualWorkspaceSize{0};				// attempt to optimize memory allocated
	cutensorPlanGetAttribute(globalHandle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize));

    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        cudaMalloc(&work, actualWorkspaceSize);
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

	/*
	 *	Step 3: Actual execution
	 *
	 *
	 * */

	cudaStream_t stream;
	cudaStreamCreate(&stream);

  	double minTimeCUTENSOR = 1e100;
  	
	for (int i = 0; i < 3; ++i) 
	{
    		cudaMemcpy(C.Tensor::getm_pDevice(), C, C.Tensor::getm_byteSize(), cudaMemcpyHostToDevice);
    		cudaDeviceSynchronize();

    		GPUTimer timer;
    		timer.start();

    		cutensorContract(handle, 
		       plan, 
		       (void *)&alpha, A.Tensor::getm_pDevice(), B.Tensor::getm_pDevice(),
		       (void *)&beta, C.Tensor::getm_pDevice(), C.Tensor::getm_pDevice(), 
		       work, actualWorkspaceSize, stream);

    		auto time = timer.seconds();
    		minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
  	}

  

  	double transferedBytes = C.get_mbyteSize() + A.get_mbyteSize() + B.get_mbyteSize();
  	transferedBytes += ((float)beta != 0.f) ? sizeC : 0;
  	transferedBytes /= 1e9;
	
  	printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", 
	  gflops / minTimeCUTENSOR,
	  transferedBytes / minTimeCUTENSOR);

	// TODO: Free memory!
	
	return C;
}


