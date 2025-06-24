#include "../include/Index.hpp"
#include "../include/Tensor.hpp"
#include "../include/CudaUtils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <iostream>

// Globals =>

cutensorHandle_t globalHandle{nullptr};
cusolverDnHandle_t solverHandle{nullptr};
const uint32_t kAlignment{128};

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

		this->setZero();
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
	m_indices.reserve(m_extents.size());
	for (const auto& i : extents)
	{
		m_elements *= i;
		m_indices.emplace_back(Index(i));
	}

	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this->initOnHost();
		this->initOnDevice();

		this->setZero();			
	}
	
	cutensorCreateTensorDescriptor(globalHandle,			
				  &(this->m_desc),				
				  m_order,
				  m_extents.data(),		
				  NULL,		
				  CUTENSOR_R_32F,
				  kAlignment);
}

Tensor::~Tensor() = default;

// Memory Management =>
void Tensor::initOnHost()
{
	m_pHost = std::make_unique<floatType[]>(m_elements);
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
int Tensor::getOrder() const {return this->m_order;}
size_t Tensor::getElements() const {return this->m_elements;}
size_t Tensor::getByteSize() const {return this->m_byteSize;}

floatType* Tensor::getHostPtr() const {return this->m_pHost.get();}			// WARN: returns float*

cutensorTensorDescriptor_t Tensor::getDesc() const {return this->m_desc;}
void* Tensor::getDevicePtr() const {return this->m_pDevice;}	

// Set values of the Tensor =>
void Tensor::setInt(const int val)
{ 
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
	for(size_t j = 0; j < m_elements; ++j)						// populate the tensor
	{
		m_pHost[j] = ((floatType) rand())/RAND_MAX;
	}
	this->cpyToDevice();
}

// Basic unary Operations [call A.operation();] =>	
floatType Tensor::fNorm()
{
	return std::sqrt(this->fNormSquared());
}		
floatType Tensor::fNormSquared()
{
	floatType res{0};
	for(int i = 0; i < m_elements; ++i)
	{
		res+= m_pHost[i] * m_pHost[i];
	}
	return res;
}
void Tensor::primeAll()
{
	for(int i = 0; i < m_order; ++i)
	{
		m_indices[i].prime(m_modes[i]);
	}
}
void Tensor::nextPermute()
{

}

void Tensor::matchPermute(const Tensor& other)
{
	
} 

// TODO: Operator overloads go here =>

// Tensor Operations =>

// flatten is a helper function that reshapes a tensor to a matrix for operations on matrices like svd(), qr() etc.
// It does NOT change the m_modes or m_indices of the tensor themselves and it is expected that a flatten
// call will be paired with an unflatten call if the tensor needs to be used later. 
// unflatten will liberally use the original tensor m_indices or m_modes to reconstruct the tensor to its correct shape
void Tensor::flatten(int split)
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
		strides = {tmp2, 1LL};
		m_extents = {tmp1, tmp2};
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
		m_extents = {tmp1};
		m_order = 1;
	}
}

void Tensor::unflatten(const std::vector<int64_t>& targetExtents, int targetOrder)
{
	m_extents = targetExtents;
	m_order = targetOrder;
}

std::tuple<Tensor, Tensor, Tensor> Tensor::svd(int split)
{
	std::vector<int64_t> copyExtents = m_extents;
	int copyOrder = m_order;
	flatten(split);
	this->cpyToDevice();
	
	int m = m_extents[0], n = m_extents[1];
	int k = std::min(m, n);

	Index indsU(m);
	Index indsS(k);
	Index indsVd(n);

	Tensor U({indsU, indsS}), S({indsS}), Vd({indsS, indsU});

	int info = 0;
	int* devInfo{nullptr};
	int lwork{0};

	cudaStream_t stream = NULL;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    	cusolverDnSetStream(solverHandle, stream);

	gesvdjInfo_t gesvdj_params = NULL;
	const double tol = 1.e-7;
    	const int max_sweeps = 15;

	cusolverDnCreateGesvdjInfo(&gesvdj_params);
	cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
	cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);

	cusolverDnSgesvdj_bufferSize(solverHandle,
			      CUSOLVER_EIG_MODE_NOVECTOR,
			      1, 
			      m, n,
			      reinterpret_cast<floatTypeCompute*>(this->m_pDevice), m,
			      reinterpret_cast<floatTypeCompute*>(S.m_pDevice),
			      reinterpret_cast<floatTypeCompute*>(U.m_pDevice), m,   // U: m×k, lda=m
			      reinterpret_cast<floatTypeCompute*>(Vd.m_pDevice), k,  // Vᵀ: k×n, ldv=k
			      &lwork,
			      gesvdj_params);
	
	floatTypeCompute *workDevice = nullptr;
	cudaMalloc(reinterpret_cast<void **>(&workDevice), sizeof(floatTypeCompute) * lwork);
	
	cusolverDnSgesvdj(solverHandle,
		   CUSOLVER_EIG_MODE_NOVECTOR,
		   1,
		   m, n,
		   reinterpret_cast<floatTypeCompute*>(this->m_pDevice), m,
		   reinterpret_cast<floatTypeCompute*>(S.m_pDevice),
		   reinterpret_cast<floatTypeCompute*>(U.m_pDevice), m,
		   reinterpret_cast<floatTypeCompute*>(Vd.m_pDevice), k,
		   workDevice, lwork,
		   devInfo, gesvdj_params);

    	cudaStreamSynchronize(stream);
    	{
      		cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
      		assert(info == 0 && "SVD did not converge");
    	}
	
	U.cpyToHost();
	S.cpyToHost();
	Vd.cpyToHost();

    	cudaFree(workDevice);
    	cudaFree(devInfo);
	
	unflatten(copyExtents, copyOrder);
	return std::make_tuple(std::move(U), std::move(S), std::move(Vd));
}

Tensor contractAB(const Tensor& A, const Tensor& B)
{
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

	auto[modesC, extentsC] = getUniqueIndsAB(A, B);

	// C only needs its (IDs, dims) for our purposes, so this initialization will do
	Tensor C(modesC, extentsC);


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
			   C.getDesc(), C.getModes().data(),				// Output to D	
			   descCompute);

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

	cudaStream_t stream;
	cudaStreamCreate(&stream);
    	cudaStreamSynchronize(stream);

	floatTypeCompute alpha{1.0f}, beta{0.0f};
    	cutensorContract(globalHandle, 
		      plan, 
		      (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
		      (void *)&beta, C.getDevicePtr(), C.getDevicePtr(), 
		      work, workspaceSize, stream);
	
	// Can add a CPU routine before the sync which forces the CPU to wait for the GPU to finish work

	cudaStreamSynchronize(stream);
	C.cpyToHost();

	// note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr into our host ptr

    	// DONE:  TODO: Free memory!
	cudaFree(work);
	cutensorDestroyPlan(plan);
	cudaStreamDestroy(stream);
	cutensorDestroyOperationDescriptor(descOp);

	std::cout << C.getElements();
	for(int i = 0; i < C.getElements(); ++i)
	{
     		std::cout <<  C.getHostPtr()[i] << " ";
	}

	return C;	
}

// Judiciously use the elementwise + operators if addition is the main goal, the contract function above will be 
// faster since it doesn't allocate resources for an extra tensor D. Providing the functionality as cuTENSOR natively
// provides it as well. 

Tensor axpyABC(const Tensor& A, const Tensor& B, const Tensor& C)
{
	return axpyABC(1.0f, A, B, 1.0f, C);
}

Tensor axpyABC(floatType alpha, const Tensor& A, const Tensor& B, floatType beta, const Tensor& C)
{
// Initialize D; note that D has the same modes, extents as D, (but it will be set to zero via our initializer)
	Tensor D(C.getModes(), C.getExtents());

  	cutensorOperationDescriptor_t descOp;			
	cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;	

	cutensorCreateContraction(globalHandle,
			   &descOp,
			   A.getDesc(), A.getModes().data(), CUTENSOR_OP_IDENTITY,	
			   B.getDesc(), B.getModes().data(), CUTENSOR_OP_IDENTITY,
			   C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY,
			   D.getDesc(), D.getModes().data(),				// Output to D	
			   descCompute);

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

	cudaStream_t stream;
	cudaStreamCreate(&stream);
    	cudaStreamSynchronize(stream);

    	cutensorContract(globalHandle, 
		      plan, 
		      (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
		      (void *)&beta, C.getDevicePtr(), D.getDevicePtr(), 
		      work, workspaceSize, stream);
	
	// Can add a CPU routine before the sync which forces the CPU to wait for the GPU to finish work

	cudaStreamSynchronize(stream);
	D.cpyToHost();

	// note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr into our host ptr

    	// DONE:  TODO: Free memory!
	cudaFree(work);
	cutensorDestroyPlan(plan);
	cudaStreamDestroy(stream);
	cutensorDestroyOperationDescriptor(descOp);

	return D;	
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


