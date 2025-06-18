#pragma once

#include "../include/Index.hpp"
#include "../include/Tensor.hpp"
#include "../include/CudaUtils.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

// Globals =>

cutensorHandle_t globalHandle{nullptr};
const uint32_t kAlignment{128};

using floatType = float;
using floatTypeCompute = float;

// Constructors =>
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
	
	cutensorCreateTensorDescriptor(globalHandle,			
				  &(this->m_desc),				
				  m_order,
				  m_extents.data(),		
				  NULL,					// Stride (refer below! line[169])	
				  CUTENSOR_R_32F,			// Datatype: 32-bit Real Floats
				  kAlignment);
}
	
Tensor::Tensor(const std::vector<int>& modes, const std::vector<int64_t>& extents)	// alternate ctor
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
	
	cutensorCreateTensorDescriptor(globalHandle,			
				  &(this->m_desc),				
				  m_order,
				  m_extents.data(),		
				  NULL,		
				  CUTENSOR_R_32F,
				  kAlignment);
}

// Memory Management =>
void Tensor::initOnHost()
{
	m_pHost = std::make_unique<floatType[]>(m_byteSize);
}

void Tensor::initOnDevice()
{
	cudaMalloc((void**)& (m_pDevice), m_byteSize);
}

void Tensor::freeMemory()
{
	m_pHost.reset();
	if(m_pDevice)
	{
		cudaFree(m_pDevice);
		m_pDevice = nullptr;
	}
}

void Tensor::cpyToDevice() const
{
	cudaMemcpy(m_pDevice, m_pHost.get(), m_byteSize, cudaMemcpyHostToDevice);		// copy tensor to GPU
}

void Tensor::cpyToHost() const
{
	cudaMemcpy(m_pHost.get(), m_pDevice, m_byteSize, cudaMemcpyDeviceToHost);		// copy tensor to Host
}

// Getters =>
const std::vector<Index>& Tensor::getInds() const {return this->m_indices;}
const std::vector<int>& Tensor::getModes() const {return this->m_modes;}
const std::vector<int64_t>& Tensor::getExtents() const {return this->m_extents;}

int Tensor::getOrder() const	{return this->m_order;}
size_t Tensor::getElements() const {return this->m_elements;}
size_t Tensor::getByteSize() const {return this->m_byteSize;}

floatType* Tensor::getHostPtr() const {return this->m_pHost.get();}

cutensorTensorDescriptor_t Tensor::getDesc() const {return this->m_desc;}
void* Tensor::getDevicePtr() const {return this->m_pDevice;}	

// Set values of the Tensor =>
void Tensor::setInt(const int val)
{ 
	this->freeMemory();
	initOnHost();
	initOnDevice();

	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = val;
	}
	this->cpyToDevice();
}

void Tensor::setZero()
{
	Tensor::setInt(0);
}

void Tensor::setOne()
{	
	Tensor::setInt(1);
}

void Tensor::setRand()
{		
	this->freeMemory();
	initOnHost();
	initOnDevice();

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

// We now define the parameter split, as the integer l, such that all indices upto and including the l-th index, 
// becomes the new row index, and the l+1-th to the N-th index becomes the column index; for our most useful case
// of reshaping a tensor into a matrix.

	std::vector<int64_t> strides;

	if (split < m_order && split > 0) 			// if split == m_order or 0, then we have a rank-1
	{
		int tmp1{1}, tmp2{1};
		for (int i = 0; i < split; ++i)
		{
			tmp1 *= m_extents[i];
		}
		for (int i = split; i < m_order; ++i)
		{
			tmp2 *= m_extents[i]; 
		}	
		strides = {static_cast<int64_t>(tmp2), 1LL};
		m_extents = {static_cast<int64_t>(tmp1), static_cast<int64_t>(tmp2)};
		m_order = 2;
	}

// And for the case of reshaping to a rank-1 tensor (grouping all indices)

	else if (split == m_order || split == 1)
	{
		int tmp1{1};
		for(const auto& i: m_extents)
		{
			tmp1 *= i;
		}
		strides = {1LL};
		m_extents = {static_cast<int64_t>(tmp1)};
		m_order = 1;
	}
		
	cutensorCreateTensorDescriptor(globalHandle,
				&(this->m_desc),
				m_order, 
				m_extents.data(),
				strides.data(),
				CUTENSOR_R_32F,
				kAlignment);
}

Tensor contractAB(const Tensor& A, const Tensor& B)
{
	/* 
	 *	The contraction boilerplate is broken down into 3 major steps
	 *
	 * 	Step 1: Describe the tensors in a suitable format 
	*/
	// We assume A and B have called initOnHost, some value setting function and initOnDevice
	if(A.getHostPtr() && A.getDevicePtr())
	{
		A.cpyToDevice();
	}
	else 
	{
		// TODO: Manage the error
		throw std::runtime_error("Tensor A has invalid memory pointers!");
	}

	if(B.getHostPtr() && B.getDevicePtr())
	{
		B.cpyToDevice();
	}
	else
	{
		throw std::runtime_error("Tensor B has invalid memory pointers!");
	}

	// DONE: TODO: Create a lookup table (IDs, dims) to initialize indices left in the output Tensor C
	std::pair<std::vector<int>, std::vector<int64_t>> initC = getUniqueIndsAB(A, B);

	// C only needs its (IDs, dims) for our purposes, so this initialization will do
	Tensor C(initC.first, initC.second);
	C.initOnHost();
	C.setZero();
	C.initOnDevice();

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
			   A.getDesc(), A.getModes().data(), CUTENSOR_OP_IDENTITY,	// descA, A.m_modes, opA
			   B.getDesc(), B.getModes().data(), CUTENSOR_OP_IDENTITY,
			   C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY,
			   C.getDesc(), C.getModes().data(),				// Output to C	
			   descCompute);
	
  	floatTypeCompute alpha = (floatTypeCompute)1.0f;
  	floatTypeCompute beta = (floatTypeCompute)0.f;

	cutensorPlan_t plan = makeContractionPlan(descOp);

	uint64_t workspaceSize{0};				// attempt to optimize memory allocated
	cutensorPlanGetAttribute(globalHandle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &workspaceSize,
                sizeof(workspaceSize));

    	void *work = nullptr;
    	if (workspaceSize > 0)
    	{
        	cudaMalloc(&work, workspaceSize);
        	assert(uintptr_t(work) % 128 == 0); 		// workspace must be aligned to 128 byte-boundary
    	}

	/*
	 *	Step 3: Actual execution
	*/

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaMemcpy(C.getDevicePtr(), C.getHostPtr(), C.getByteSize(), cudaMemcpyHostToDevice);
    	cudaDeviceSynchronize();

    	cutensorContract(globalHandle, 
		      plan, 
		      (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
		      (void *)&beta, C.getDevicePtr(), C.getDevicePtr(), 
		      work, workspaceSize, stream);
	
	// Can add a CPU routine before the sync which forces the CPU to wait for the GPU to finish work

	cudaStreamSynchronize(stream);
	cudaMemcpy(C.getHostPtr(), C.getDevicePtr(), C.getByteSize(), cudaMemcpyDeviceToHost);	
	// note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr into our host ptr

    	// DONE:  TODO: Free memory!
	cudaFree(work);
	cutensorDestroyPlan(plan);
	cudaStreamDestroy(stream);
	cutensorDestroyOperationDescriptor(descOp);

	return C;	
}
// TODO: Other Contract overloads; where we just prime the indices that fall into getUniqueIndsAB but the user
// does not want to contract, then pass the changed Tensors to contractAB

// Helper function to figure out the indices of C = A * B, 
// returns a pair of vectors (modesC, extentsC) that we assign to (C.m_modes, C.m_extents) =>
std::pair<std::vector<int>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B)
{
 	// Builds a map of (B.modes, B.extents), (necessary to enforce the order A, B)
	std::unordered_map<int,int64_t> mapB;

	// Since getUniqueIndsAB is not a member function we extract what we need at the start
	std::vector<int> modesA = A.getModes();
	std::vector<int> modesB = B.getModes();
	std::vector<int64_t> extentsA = A.getExtents();
	std::vector<int64_t> extentsB = B.getExtents();

	for (size_t i = 0; i < modesB.size(); ++i) 
	{
        	mapB.emplace(modesB[i], extentsB[i]);
    	}

	std::vector<std::pair<int,int64_t>> tmp;
    	tmp.reserve(modesA.size() + modesB.size());
	
	/* 
	 * Note that this is a vector of pairs, purely because it makes more sense logically to find the data
	 * (modes_i, extents_i) for a particular unique Index_i than to find all modes, then all extents in 
	 * necessarily the same order
	*/

    	// Adds those in A, not in B
    	for (size_t j = 0; j < modesA.size(); ++j) 
	{
        	int mode = modesA[j];
        	auto it = mapB.find(mode);
        	if (it == mapB.end()) 
		{
            		tmp.emplace_back(mode, extentsA[j]);			// Builds the pair in place
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
	std::vector<int> modesC;
    	std::vector<int64_t> extentsC;
    	modesC.reserve(tmp.size());
    	extentsC.reserve(tmp.size());
    	for (auto &pr : tmp) {
        	modesC.push_back(pr.first);
        	extentsC.push_back(pr.second);
    	}

    	return { std::move(modesC), std::move(extentsC) };
}


