#pragma once

#include "../include/Index.hpp"
#include "../include/Tensor.hpp"
#include "../include/utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <unordered_map>
#include <vector>

// Globals =>
cutensorHandle_t globalHandle{nullptr};
const uint32_t kAlignment{128};

using floatType = float;

// Constructors =>
Tensor::Tensor() = default; 

// DONE:  TODO: Move the logic outside the .hpp 

Tensor::Tensor(const std::vector<Index>& indices)
: m_indices(indices), m_order(indices.size()), m_elements(1)
{
	m_modes.resize(m_order);
	m_extents.resize(m_order);
	for (size_t i = 0; i < m_order; ++i)
	{
		m_modes[i] = indices[i].getUniqueID();
		m_extents[i] = indices[i].getDim();
		m_elements *= m_extents[i];
	}
	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this->initOnHost();
		this->initOnDevice();

		this->setRand();
	}
	
	cutensorCreateTensorDescriptor_t(globalHandle,			
				  &(this->m_desc),				
				  m_order,
				  m_extents,		
				  NULL,					// Stride (refer below! line[196])		
				  CUTENSOR_R_32F,			// Datatype: 32-bit Real Floats
				  kAlignment);
}



Tensor::Tensor(const std::map<size_t, int64_t>& lookup)		// lookup = {(key,value): (Index id, Index dim)}
: m_indices({0}), m_order(lookup.size()), m_elements(1)
{
	m_modes.resize(m_order);
	m_extents.resize(m_order);
	for (const auto& [id, dim]: lookup)
	{
		m_modes.push_back(id);
		m_extents.push_back(dim); 
		m_elements *= dim;
	}
	
	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this->initOnHost();
		this->initOnDevice();

		this->setRand();			
	}
	
	cutensorCreateTensorDescriptor_t(globalHandle,			
				  &(this->m_desc),				
				  m_order,
				  m_extents,		
				  NULL,				
				  CUTENSOR_R_32F,	
				  kAlignment);
}
	
Tensor::Tensor(const std::vector<size_t>& modes, const std::vector<int64_t>& extents)	// alternate ctor
: m_modes(modes), m_extents(extents), m_order(modes.size()), m_elements(1)
{
	for (const auto& i : extents)
	{
		m_elements *= i;
	}

	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this->initOnHost();
		this->initOnDevice();

		this->setRand();			
	}
	
	cutensorCreateTensorDescriptor_t(globalHandle,			
				  &(this->m_desc),				
				  m_order,
				  m_extents,		
				  NULL,		
				  CUTENSOR_R_32F,
				  kAlignment);
}

// Getters =>
const std::vector<Index>& Tensor::getInds() const {return this->m_indices;}
const std::vector<size_t>& Tensor::getModes() const {return this->m_modes;}
const std::vector<int64_t>& Tensor::getExtents() const {return this->m_extents;}

size_t Tensor::getOrder() const	{return this->m_order;}
size_t Tensor::getElements() const {return this->m_elements;}
size_t Tensor::getByteSize() const {return this->m_byteSize;}

floatType* Tensor::getHostPtr() const {return this->m_pHost;}

cutensorTensorDescriptor_t Tensor::getDesc() const {return this->m_desc;}
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
void Tensor::cpyToDevice()
{
	cudaMemcpy(m_pDevice, m_pHost, m_byteSize, cudaMemcpyHostToDevice);		// copy tensor to GPU
}

// Set values of the Tensor =>
void Tensor::setZero()
{
	this->freeMemory();
	m_pHost = (floatType*) malloc(m_byteSize);
	cudaMalloc((void**)& m_pDevice, m_byteSize);
			
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = 0;
	}
	
	this->cpyToDevice();
}

void Tensor::setOne()
{	
	this->freeMemory();
	m_pHost = (floatType*) malloc(m_byteSize);
	cudaMalloc((void**)& m_pDevice, m_byteSize);
		
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = 1;
	}
	this->cpyToDevice();
}

void Tensor::setRand()
{		
	this->freeMemory();
	m_pHost = (floatType*) malloc(m_byteSize);
	cudaMalloc((void**)& m_pDevice, m_byteSize);
		
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = ((floatType) rand())/RAND_MAX;
	}
	this->cpyToDevice();
}

// Tensor Operations =>


void Tensor::reshape(int split)
{ 
/*
 * A comment about stride:
 * 	Let's say we have a tensor A with the indices i, j, k => dim(i) = 3, dim(j) = 4, dim(k) = 5; so the order N = 3
 *	We lay these out as, (in the so-called row-major form)
 *	[0, 0, 0]
 *	[0, 0, 1] => the last index k, updates the fastest!
 *	[0, 0, 2]
 *	[0, 0, 3]
 *	[0, 0, 4]
 *	[0, 1, 0] => the second-to-last index j updates every 5 (dim(k)) entries;
 *	[0, 1, 1]    We can extrapolate that i will update after every 5 * 4 = 20 (dim(k) * dim(j)) entries 
 *	.	     And in general the l-th index will update after every dim(l+1) * dim(l+2) ... dim(N) entries
 *	.
 *	.
 *	[2, 3, 4]
 *	
 *	To reshape these Tensors into matrices, say A, we group some indices to be the row_indices, and some to be the
 *	column_indices; let's say we group (i,j), and let k be separate. 
 *	k still updates as before, with a 'stride' of 1. Since we need to reshape (i, j) into one unified index, we
 *	see that it will update with the old stride of the right-most index inside it, j, which means (i, j) now has 
 *	stride 5.
 *	This means the vector strides, which was {20, 5, 1} before, will now change to {5, 1}.
 *	Note that our extents and orders will also change as we need to initialize two new indices (in this case
 *	k can remain as is). Our extents will go from {3, 4, 5} to {12, 5}; note that the "volume" remains unchanged.
 *	To reshape our tensor, we need to simply change the strides, extents vectors and the order in 
 *	cutensorCreateDescriptor_t, and change the m_desc.
*/

// We now define the parameter break, as the integer l, such that all indices upto and including the l-th index, 
// becomes the new row index, and the l+1-th to the N-th index becomes the column index; for our most useful case
// of reshaping a tensor into a matrix.
	
	if (split < m_order && split > 0) 			// if split == m_order or 0, then we have a vector
	{
		size_t tmp1{1}, tmp2{1};
		for (int i = 0; i < break; ++i)
		{
			tmp1 *= m_extents[i];
		}
		for (int i = break; i < m_order; ++i)
		{
			tmp2 *= m_extents[i]; 
		}	
		std::vector<size_t> strides = {tmp2, 1};
		std::vector<size_t> m_extentsNew = {tmp1, tmp2};
		m_extents = m_extentsNew;
		m_order = 2;

		cutensorCreateTensorDescriptor_t(globalHandle,
				   &(this->m_desc),
				   m_order,
				   m_extents,
				   strides,
				   CUTENSOR_R_32F,
				   kAlignment);
	}
}

Tensor contractAB(const Tensor& A, const Tensor& B)
{
	/* 
	 *	The contraction boilerplate is broken down into 3 major steps
	 *
	 * 	Step 1: Describe the tensors in a suitable format 
	*/
	
	// DONE: TODO: Create a lookup table (IDs, dims) to initialize indices left in the output Tensor C
	std::pair<std::vector<size_t>, std::vector<int64_t>> initC = getUniqueIndsAB(A, B);

	// C only needs its (IDs, dims) for our purposes, so this initialization will do
	Tensor::Tensor C(initC.first, initC.second);

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
			   A.getDesc(), A.getModes(), CUTENSOR_OP_IDENTITY,	// descA, A.m_modes, opA
			   B.getDesc(), B.getModes(), CUTENSOR_OP_IDENTITY,
			   C.getDesc(), C.getModes(), CUTENSOR_OP_IDENTITY,
			   C.getDesc(), C.getModes(),				// Output to C	
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

	cudaMemcpy(C.getDevicePtr(), C.getHostPtr(), C.getByteSize(), cudaMemcpyHostToDevice);
    	cudaDeviceSynchronize();

    	cutensorContract(handle, 
		      plan, 
		      (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
		      (void *)&beta, C.getDevicePtr(), C.getDevicePtr(), 
		      work, actualWorkspaceSize, stream);

    	// TODO: Free memory!
	
	return C;	
}

// TODO: Other Contract overloads; where we just prime the indices that fall into getUniqueIndsAB but the user
// 	 does not want to contract, then pass the changed Tensors to contractAB

// Helper function to figure out the indices of C = A * B, 
// returns a pair of vectors (modesC, extentsC) that we assign to (C.m_modes, C.m_extents) =>
std::pair<std::vector<size_t>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B)
{
 	// Builds a map of (B.modes, B.extents), (necessary to enforce the order A, B)
	std::unordered_map<size_t,int64_t> mapB;
	mapB.resize(B.m_modes.size());
	for (size_t i = 0; i < B.m_modes.size(); ++i) 
	{
        	mapB.emplace(B.m_modes[i], B.m_extents[i]);
    	}

	std::vector<std::pair<size_t,int64_t>> tmp;
    	tmp.reserve(A.m_modes.size() + B.m_modes.size());
	
	/* 
	 * Note that this is a vector of pairs, purely because it makes more sense logically to find the data
	 * (modes_i, extents_i) for a particular unique Index_i than to find all modes, then all extents in 
	 * necessarily the same order
	*/

    	// Adds those in A, not in B
    	for (size_t j = 0; j < A.m_modes.size(); ++j) 
	{
        	size_t mode = A.m_modes[j];
        	auto it = mapB.find(mode);
        	if (it == mapB.end()) 
		{
            		tmp.emplace_back(mode, A.m_extents[j]);			// Builds the pair in place
        	} 
		else 
		{
            		// common IDs get erased
            		mapB.erase(it);
        	}
    	}

    	// Adds those in B, not in A (common IDs have already been erased)
    	for (auto &p : mapB) {
        	tmp.emplace_back(p.first, p.second);
    	}
	
	// Since for us it is more useful to have a pair of two separately contiguous vectors in the Tensor object,
	// we eventually return a pair<vector<size_t>, vector<int64_t>>, i.e (C.m_modes, C.m_extents)
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


