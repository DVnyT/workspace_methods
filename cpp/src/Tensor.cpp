#pragma once

#include "../include/Tensor.hpp"
#include "../include/Index.hpp"
#include "../include/utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <unordered_map>
#include <vector>

// Constructors =>
Tensor::Tensor() = default; 

// DONE:  TODO: Move the logic outside the .hpp 

Tensor::Tensor(const std::vector<Index>& indices)
: m_indices(indices), m_order(indices.size())
{
	m_modes.reserve(m_order);
	m_extents.reserve(m_order);
	for (size_t i = 0; i < m_order; ++i)
	{
		m_modes[i] = indices[i].getUniqueID();
		m_extents[i] = indices[i].getDim();
		m_elements *= m_extents[i];
	}
	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this -> setRand();				// setRand() also initializes m_pHost, m_pDevice
	}
}

Tensor::Tensor(const std::map<size_t, int64_t>& lookup)		// lookup = {(key,value): (Index id, Index dim)}
: m_indices({0}), m_order(lookup.size())
{
	m_modes.reserve(m_order);
	m_extents.reserve(m_order);
	for (const auto& [id, dim]: lookup)
	{
		m_modes.push_back(id);
		m_extents.push_back(dim); 
		m_elements *= dim;
	}
	m_byteSize = m_elements * sizeof(float);
	
	if (m_byteSize != 0) 
	{
		this -> setRand();
	}
}
	
Tensor::Tensor(const std::vector<size_t>& modes, const std::vector<int64_t>& extents)	// alternate ctor
: m_modes(modes), m_extents(extents), m_order(modes.size())
{
	for (const auto& i : extents)
	{
		m_elements *= i;
	}
		
	m_byteSize = m_elements * sizeof(float);
	
	if (m_byteSize != 0) 
	{
		this -> setRand();
	}
}

// Getters =>
const std::vector<size_t>& Tensor::getModes() const {return this->m_modes;}
const Tensor::std::vector<int64_t>& Tensor::getExtents() const {return this->m_extents;}
size_t Tensor::getOrder() const	{return this->m_order;}
size_t Tensor::getElements() const {return this->m_elements;}
size_t Tensor::getByteSize() const {return this->m_byteSize;}
float* Tensor::getHostPtr() const {return this->m_pHost;}
void* Tensor::getDevicePtr() const {return this->m_pDevice;}	

// Memory Management =>
void Tensor::freeMemory()
{
	if(m_pHost)
	{
		free(m_pHost);
		m_pHost = nullptr;
	}
	if(m_pDevice)
	{
		cudaFree(m_pDevice);
		m_pDevice = nullptr;
	}
}

// Set values of the Tensor =>
void Tensor::setZero()
{
	this->freeMemory();
	m_pHost = (float*) malloc(m_byteSize);
	cudaMalloc((void**)& m_pDevice, m_byteSize);
		
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = 0;
	}

	cudaMemcpy(m_pDevice, m_pHost, m_byteSize, cudaMemcpyHostToDevice);		// copy tensor to GPU
}

void Tensor::setOne()
{	
	this->freeMemory();
	m_pHost = (float*) malloc(m_byteSize);
	cudaMalloc((void**)& m_pDevice, m_byteSize);
		
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = 1;
	}

	cudaMemcpy(m_pDevice, m_pHost, m_byteSize, cudaMemcpyHostToDevice);		// copy tensor to GPU
}

void Tensor::setRand()
{		
	this->freeMemory();
	m_pHost = (float*) malloc(m_byteSize);
	cudaMalloc((void**)& m_pDevice, m_byteSize);
		
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = ((float) rand())/RAND_MAX;
	}

	cudaMemcpy(m_pDevice, m_pHost, m_byteSize, cudaMemcpyHostToDevice);		// copy tensor to GPU
}

// Tensor Operations =>
void Tensor::reshape(const std::vector<Index::Index>& column_Indices, 
		     const std::vector<Index::Index>& row_Indices)
{

}
Tensor contractAB(const Tensor::Tensor& A, 
		const Tensor::Tensor& B,  
		std::vector<Index::Index>& toContract,
		cutensorHandle_t& globalHandle)
{
	/* 
	 *	The contraction boilerplate is broken down into 3 major steps
	 *
	 * 	Step 1: Describe the tensors in a suitable format 
	*/

	cutensorTensorDescriptor_t descA;				// Allocated tensor descriptor for A
	cutensorTensorDescriptor_t descB;
	
	// DONE: TODO: Create a lookup table (IDs, dims) to initialize indices left in the output Tensor C
	std::pair<std::vector<size_t>, std::vector<int64_t>> initC = getUniqueIndsAB(A, B);
	
	// C only needs its (IDs, dims) for our purposes, so this initialization will do
	Tensor::Tensor C = new(Tensor::Tensor(initC.first, initC.second));

	const uint32_t kAlignment = 128;  				// TODO: Do make this a global variable!
		
	cutensorCreateTensorDescriptor_t(globalHandle,			
				  &descA,				
				  A.getOrder(),
				  A.getExtents(),		
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
				  B.getOrder(),
				  B.getExtents(),
				  NULL,
				  CUTENSOR_R_32F,
				  kAlignment);
	

	cutensorCreateTensorDescriptor_t(globalHandle,			// Output Tensor C for a simple contraction
				  &descC,
				  C.getOrder(),
				  C.getExtents(),
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
	*	output Tensor D. Note that modes_D ≡ modes_C ≡ modes_(A * B).
	*	
	*	opA() does an operation on all elements of A, we currently use the identity operation.
	*/
	
  	cutensorOperationDescriptor_t descOp;			// will encode the operation, init. by function below
	cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;	// Precision of contraction

	cutensorCreateContraction(globalHandle,
			   &descOp,
			   descA, A.getModes(), CUTENSOR_OP_IDENTITY,	// descA, A.m_modes, opA
			   descB, B.getModes(), CUTENSOR_OP_IDENTITY,
			   descC, C.getModes(), CUTENSOR_OP_IDENTITY,
			   descC, C.getModes(),				// Output to C	
			   descCompute);
	
  	typedef float floatTypeCompute;
  	floatTypeCompute alpha = (floatTypeCompute)1.0f;
  	floatTypeCompute beta = (floatTypeCompute)0.f;

	cutensorPlan_t plan;
	cutensorPlanPreference_t planPref;					
	// plan holds the work to be done, in this case a contraction, along with all the parameters we tune now
	// planPref narrows the choices of algorithms/variants/kernels to use
  	
	const cutensorAlgo_t algo = chooseContractionAlgo();			// default is CUTENSOR_ALGO_DEFAULT
	
	cutensorCreatePlanPreference(globalHandle, 
				&planPref, 					// mode and algo go into planPref
				algo,
				CUTENSOR_JIT_MODE_NONE);			// Not using Just-In-Time compilation

	uint64_t workspaceSizeEstimate{0};					// outputs estimated size to this
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
        	assert(uintptr_t(work) % 128 == 0); 		// workspace must be aligned to 128 byte-boundary
    	}

	/*
	 *	Step 3: Actual execution
	 *
	 * */

	cudaStream_t stream;
	cudaStreamCreate(&stream);

  	double minTimeCUTENSOR = 1e100;
  	
	for (int i = 0; i < 3; ++i) 
	{
    		cudaMemcpy(C.getDevicePtr(), C, C.getByteSize(), cudaMemcpyHostToDevice);
    		cudaDeviceSynchronize();

    		GPUTimer timer;
    		timer.start();

    		cutensorContract(handle, 
		       plan, 
		       (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
		       (void *)&beta, C.getDevicePtr(), C.getDevicePtr(), 
		       work, actualWorkspaceSize, stream);

    		auto time = timer.seconds();
    		minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
  	}

  	double transferedBytes = C.getByteSize() + A.getByteSize() + B.getByteSize();
  	transferedBytes += ((float)beta != 0.f) ? C.getByteSize() : 0;
  	transferedBytes /= 1e9;
	
	// TODO: Free memory!
	
	return C;	
}

// Helper function to figure out the indices of C = A * B, 
// returns a pair of vectors (modesC, extentsC) that we assign to (C.m_modes, C.m_extents) =>
std::pair<std::vector<size_t>, std::vector<int64_t>> getUniqueInds_AB(const Tensor& A, const Tensor& B)
{
 	// Builds a map of (A.modes, A.extents)
	std::unordered_map<size_t,int64_t> mapA;
	mapA.reserve(A.m_modes.size());
	for (size_t i = 0; i < A.m_modes.size(); ++i) 
	{
        	mapA.emplace(A.m_modes[i], A.m_extents[i]);
    	}

	std::vector<std::pair<size_t,int64_t>> tmp;
    	tmp.reserve(A.m_modes.size() + B.m_modes.size());

    	// Adds those in B, not in A
    	for (size_t j = 0; j < B.m_modes.size(); ++j) 
	{
        	size_t mode = B.m_modes[j];
        	auto it = mapA.find(mode);
        	if (it == mapA.end()) 
		{
            		tmp.emplace_back(mode, B.m_extents[j]);
        	} 
		else 
		{
            		// common IDs get erased
            		mapA.erase(it);
        	}
    	}

    	// Adds those in A, not in B (common IDs have already been erased)
    	for (auto &p : mapA) {
        	tmp.emplace_back(p.first, p.second);
    	}

	std::vector<size_t> modesC;
    	std::vector<int64_t> extentsC;
    	modesC.reserve(tmp.size());
    	extentsC.reserve(tmp.size());
    	for (auto &pr : tmp) {
        	modesC.push_back(pr.first);
        	extentsC.push_back(pr.second);
    	}

    	return { std::move(modesC), std::move(extentsC) };
}


