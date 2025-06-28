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
#include <sstream>    // for std::ostringstream
#include <iomanip>
#include <iostream>

// Globals =>
static void printLevel(std::ostream& os,
                       const floatType* data,
                       const std::vector<int64_t>& extents,
                       const std::vector<size_t>& strides,
                       int dim,
                       size_t offset)
{
    os << "[";
    int64_t N = extents[dim];
    for (int64_t i = 0; i < N; ++i) {
        size_t idx = offset + i * strides[dim];
        if (dim + 1 == (int)extents.size()) {
            // deepest level: print values
            os << data[idx];
        } else {
            // recurse into next axis
            printLevel(os, data, extents, strides, dim+1, idx);
        }
        if (i + 1 < N) os << ", ";
    }
    os << "]";
}

cutensorHandle_t globalHandle{nullptr};
cusolverDnHandle_t solverHandle{nullptr};
cublasHandle_t blasHandle{nullptr};

const uint32_t kAlignment{128};

// Constructors =>
// DONE:  TODO: Move the logic outside the .hpp 

Tensor::Tensor(const std::vector<Index>& indices)
: m_indices(indices), m_order(indices.size()), m_elements(1)
{
	m_modes.reserve(m_order);
	m_extents.reserve(m_order);
	for (size_t i = 0; i < m_order; ++i)
	{
		m_modes[i] = indices[i].getMode();
		m_extents[i] = indices[i].getExtent();
		m_elements *= m_extents[i];
	}
	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this->initOnHost();
		this->initOnDevice();
	}
	
	HANDLE_ERROR(cutensorCreateTensorDescriptor(globalHandle,			
					     &(this->m_desc),				
					     m_order,
					     m_extents.data(),		
					     NULL,			// Stride (refer below! line[169])	
					     CUTENSOR_R_32F,		// Datatype: 32-bit Real Floats
					     kAlignment));
}
	
Tensor::Tensor(const std::vector<int>& modes, const std::vector<int64_t>& extents)	// alternate ctor
: m_modes(modes), m_extents(extents), m_order(modes.size()), m_elements(1)
{
	m_indices.reserve(m_extents.size());
	for (const auto& i : extents)
	{
		m_elements *= i;
	}

	m_byteSize = sizeof(floatType) * m_elements;
	
	if(m_byteSize != 0)
	{
		this->initOnHost();
		this->initOnDevice();
	}
	
	HANDLE_ERROR(cutensorCreateTensorDescriptor(globalHandle,			
					     &(this->m_desc),				
					     m_order,
					     m_extents.data(),		
					     NULL,			// Stride (refer below! line[169])	
					     CUTENSOR_R_32F,		// Datatype: 32-bit Real Floats
					     kAlignment));
}

Tensor::~Tensor() = default;

// Memory Management =>
void Tensor::initOnHost()
{
	m_pHost = std::make_unique<floatType[]>(m_elements);
}

void Tensor::initOnDevice()
{
	HANDLE_CUDA_ERROR(cudaMalloc((void**)& (m_pDevice), m_byteSize));
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
	HANDLE_CUDA_ERROR(cudaMemcpy(m_pDevice, m_pHost.get(), m_byteSize, cudaMemcpyHostToDevice));		
	// copy tensor to GPU
}

void Tensor::cpyToHost() const
{
	HANDLE_CUDA_ERROR(cudaMemcpy(m_pHost.get(), m_pDevice, m_byteSize, cudaMemcpyDeviceToHost));		
	// copy tensor to Host
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
floatType Tensor::fNorm() const 
{
	return std::sqrt(this->fNormSquared());
}		
floatType Tensor::fNormSquared() const
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

std::ostream& operator<<(std::ostream& os, const Tensor& T) {
    // sync device → host
    T.cpyToHost();

    const auto& extents = T.getExtents();
    int order = T.getOrder();
    if (order == 0 || T.getElements() == 0) {
        os << "[ ]";
        return os;
    }

    // precompute column-major strides:
    //   stride[0] = 1
    //   stride[i] = product(extents[0..i-1])
    std::vector<size_t> strides(order, 1);
    for (int i = 1; i < order; ++i)
        strides[i] = strides[i-1] * size_t(extents[i-1]);

    // start recursion at dim 0, offset 0
    printLevel(os, T.getHostPtr(), extents, strides, 0, 0);
    return os;
}

// TODO: Operator overloads go here =>

// Tensor Operations =>

// flatten is a helper function that reshapes a tensor to a matrix for operations on matrices like svd(), qr() etc.
// It does NOT change the m_modes or m_indices of the tensor themselves and it is expected that a flatten
// call will be paired with an unflatten call if the tensor needs to be used later. 
// unflatten will liberally use the original tensor m_indices or m_modes to reconstruct the tensor to its correct shape

/*
 * A comment about stride:
 * 	Let's say we have a tensor A with the indices i, j, k => dim(i) = 3, dim(j) = 4, dim(k) = 5; so the order N = 3
 *	We lay these out as, (in column-major form)
 *	[0, 0, 0]
 *	[1, 0, 0] => the first index i, updates the fastest!
 *	[2, 0, 0]
 *	[0, 1, 0] => the second index j updates every 3 (dim(i)) entries;
 *	[1, 1, 0]    We can extrapolate that k will update after every 3 * 4 = 20 (dim(i) * dim(j)) entries 
 *	.	     And in general the l-th index will update after every dim(1) * dim(2) ... dim(l-1) entries
 *	.
 *	.
 *	[2, 3, 4]
 *	
 *	To reshape these Tensors into matrices, say A, we group some indices to be the row_indices, and some to be the
 *	column_indices; let's say we group (j, k), and let i be separate. 
 *	i still updates as before, with a 'stride' of 1. Since we need to reshape (j, k) into one unified index, we
 *	see that it will update with the old stride of the left-most index inside it, j, which means (j, k) now has 
 *	stride 3.
 *	This means the vector of strides, which was {1, 3, 12} before, will now change to {1, 3}.
 *	Note that our extents and orders will also change as we need to initialize two new indices (in this case
 *	i can remain as is). Our extents will go from {3, 4, 5} to {3, 20}; note that the "volume" remains unchanged.
 *	To reshape our tensor, we need to simply change the strides, extents vectors and the order in 
 *	cutensorCreateDescriptor_t, and change the m_desc.
*/

// We now define the parameter split, as the integer l, such that all indices upto and including the l-th index, 
// becomes the new row index, and the l+1-th to the N-th index becomes the column index; for our most useful case
// of reshaping a tensor into a matrix.

Tensor Tensor::lSVD(int split)
{
        int64_t m{1}, n{1}, k;
        int i = 0;
        if (split < m_order && split > 0) 			// if split == m_order or 0, then we have a rank-1
        {
                for (; i < split; ++i)
                {
                        m *= m_extents[i];
                }
                for (; i < m_order; ++i)
                {
                        n *= m_extents[i]; 
                }	
        }
        else
        {
                throw std::runtime_error("Unacceptable split value!");
        }
        k = std::min(m, n);
        Index indsU(m);
        Index indsS(k);
        Index indsVd(n);
        this->m_indices = {indsU, indsS};
        
        Tensor Vd(std::vector<Index>({indsS, indsVd}));
        
        size_t allocSize = sizeof(floatTypeCompute) * k;
        floatTypeCompute* SpHost = (floatTypeCompute*)malloc(allocSize);
        if (SpHost == nullptr)
        {
                throw std::runtime_error("Failed to allocate host memory for singular values!");
        }
        void* SpDevice;
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&SpDevice, allocSize));
	
	int info = 0;
        int* devInfo{nullptr};
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));
        int lwork{0};
        cudaStream_t stream = NULL;
        HANDLE_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cusolverDnSetStream(solverHandle, stream);
        
	gesvdjInfo_t gesvdj_params = NULL;
        const double tol = 1.e-7;
        const int max_sweeps = 15;
        cusolverDnCreateGesvdjInfo(&gesvdj_params);
        cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
        cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
        cusolverDnSgesvdj_bufferSize(solverHandle,
                              CUSOLVER_EIG_MODE_VECTOR,
                              1, 
                              m, n,
                              reinterpret_cast<floatTypeCompute*>(this->m_pDevice), m,
                              reinterpret_cast<floatTypeCompute*>(SpDevice),
                              reinterpret_cast<floatTypeCompute*>(this->m_pDevice), m,   // U: m×k, lda=m
                              reinterpret_cast<floatTypeCompute*>(Vd.m_pDevice), k,  // Vᵀ: k×n, ldv=k
                              &lwork,
                              gesvdj_params);
        
        floatTypeCompute *workDevice = nullptr;
        HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&workDevice), sizeof(floatTypeCompute) * lwork));
        
        cusolverDnSgesvdj(solverHandle,
                           CUSOLVER_EIG_MODE_VECTOR,
                           1,
                           m, n,
                           reinterpret_cast<floatTypeCompute*>(this->m_pDevice), m,
                           reinterpret_cast<floatTypeCompute*>(SpDevice),
                           reinterpret_cast<floatTypeCompute*>(this->m_pDevice), m,
                           reinterpret_cast<floatTypeCompute*>(Vd.m_pDevice), k,
                           workDevice, lwork,
                           devInfo, gesvdj_params);
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        {
                HANDLE_CUDA_ERROR(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
                if (info != 0)
                {
                        throw std::runtime_error("SVD did not converge! Info: " + std::to_string(info));
                }
        }
        
        this->cpyToHost();
        HANDLE_CUDA_ERROR(cudaMemcpy(SpHost, SpDevice, sizeof(floatTypeCompute) * k, cudaMemcpyDeviceToHost));
        

        HANDLE_CUDA_ERROR(cudaFree(workDevice));
        HANDLE_CUDA_ERROR(cudaFree(devInfo));
        HANDLE_CUDA_ERROR(cudaFree(SpDevice));
        HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
        cusolverDnDestroyGesvdjInfo(gesvdj_params);
        
        free(SpHost);
        
        return Vd;
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
        HANDLE_ERROR(cutensorCreateContraction(globalHandle,
                           &descOp,
                           A.getDesc(), A.getModes().data(), CUTENSOR_OP_IDENTITY,	// descA, A.m_modes, opA
                           B.getDesc(), B.getModes().data(), CUTENSOR_OP_IDENTITY,
                           C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY,
                           C.getDesc(), C.getModes().data(),				// Output to D	
                           descCompute));
        
	cutensorPlan_t plan = makeContractionPlan(descOp);
        uint64_t workspaceSize{0};				// attempt to optimize memory allocated
        HANDLE_ERROR(cutensorPlanGetAttribute(globalHandle,
                        plan,
                        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                        &workspaceSize,
                        sizeof(workspaceSize)));
        void *work = nullptr;
        if (workspaceSize > 0)
        {
                HANDLE_CUDA_ERROR(cudaMalloc(&work, workspaceSize));
                assert(uintptr_t(work) % kAlignment == 0); 	// workspace must be aligned to 128 byte-boundary
        }
        
	cudaStream_t stream;					// TODO: RAII streams!
        HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
        floatTypeCompute alpha{1.0f}, beta{0.0f};
        HANDLE_ERROR(cutensorContract(globalHandle, 
                      plan, 
                      (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
                      (void *)&beta, C.getDevicePtr(), C.getDevicePtr(), 
                      work, workspaceSize, stream));
        
        // Can add a CPU routine before the sync which forces the CPU to wait for the GPU to finish work
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        C.cpyToHost();
        // note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr into our host ptr
        // DONE:  TODO: Free memory!
        if (work)
        {
                HANDLE_CUDA_ERROR(cudaFree(work));
        }
        HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
        HANDLE_ERROR(cutensorDestroyPlan(plan));
        HANDLE_ERROR(cutensorDestroyOperationDescriptor(descOp));
       
	return C;	
}
Tensor axpyABC(const Tensor& A, const Tensor& B, const Tensor& C)
{
	return axpyABC(1.0f, A, B, 1.0f, C);
}

Tensor axpyABC(floatType alpha, const Tensor& A, const Tensor& B, floatType beta, const Tensor& C)
{
// Initialize D; note that D has the same modes, extents as D
        Tensor D(C.getModes(), C.getExtents());
        cutensorOperationDescriptor_t descOp;			
        cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;	
        HANDLE_ERROR(cutensorCreateContraction(globalHandle,
                           &descOp,
                           A.getDesc(), A.getModes().data(), CUTENSOR_OP_IDENTITY,	
                           B.getDesc(), B.getModes().data(), CUTENSOR_OP_IDENTITY,
                           C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY,
                           D.getDesc(), D.getModes().data(),				// Output to D	
                           descCompute));
        cutensorPlan_t plan = makeContractionPlan(descOp);
        uint64_t workspaceSize{0};				// attempt to optimize memory allocated
        HANDLE_ERROR(cutensorPlanGetAttribute(globalHandle,
                        plan,
                        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                        &workspaceSize,
                        sizeof(workspaceSize)));
        void *work = nullptr;
        if (workspaceSize > 0)
        {
                HANDLE_CUDA_ERROR(cudaMalloc(&work, workspaceSize));
                assert(uintptr_t(work) % 128 == 0); 		// workspace must be aligned to 128 byte-boundary
        }
        cudaStream_t stream;
        HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        HANDLE_ERROR(cutensorContract(globalHandle, 
                      plan, 
                      (void *)&alpha, A.getDevicePtr(), B.getDevicePtr(),
                      (void *)&beta, C.getDevicePtr(), D.getDevicePtr(), 
                      work, workspaceSize, stream));
        
        // Can add a CPU routine before the sync which forces the CPU to wait for the GPU to finish work
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        D.cpyToHost();
        // note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr into our host ptr
        // DONE:  TODO: Free memory!
        if (work != nullptr)
        {
                HANDLE_CUDA_ERROR(cudaFree(work));
        }
        HANDLE_ERROR(cutensorDestroyPlan(plan));
        HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
        HANDLE_ERROR(cutensorDestroyOperationDescriptor(descOp));
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


