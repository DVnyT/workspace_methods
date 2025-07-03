#include "../include/Tensor.hpp"
#include "../include/CudaUtils.hpp"
#include "../include/DevicePools.hpp"
#include "../include/Index.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream> // for std::ostringstream
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

// Globals =>
const uint32_t kAlignment{128};

// Helper Kernels =>
template <typename T> __global__ void scale_rows_kernel(T* mat, const T* vec, int rows, int cols)
{
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < rows && col < cols)
        {
                int idx = row + col * rows; // column-major
                mat[idx] *= vec[row];
        }
}

void scaleVdOnDevice(float* Vd_dev, const float* S_dev, int k, int n, cudaStream_t stream)
{
        dim3 block(32, 32);
        dim3 grid((n + 31) / 32, (k + 31) / 32);

        scale_rows_kernel<float><<<grid, block, 0, stream>>>(Vd_dev, S_dev, k, n);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
}

// Constructors =>
// DONE:  TODO: Move the logic outside the .hpp

Tensor::Tensor(const std::vector<Index>& indices) : m_indices(indices), m_order(indices.size()), m_elements(1)
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

        if (m_byteSize != 0)
        {
                this->initOnHost();
                this->initOnDevice();
        }
}

Tensor::Tensor(const std::vector<int32_t>& modes,
               const std::vector<int64_t>& extents) // alternate ctor
    : m_modes(modes), m_extents(extents), m_order(modes.size()), m_elements(1)
{
        m_indices.reserve(m_extents.size());
        for (const auto& i : extents)
        {
                m_elements *= i;
        }

        m_byteSize = sizeof(floatType) * m_elements;

        if (m_byteSize != 0)
        {
                this->initOnHost();
                this->initOnDevice();
        }
}

Tensor::~Tensor() = default;

// Memory Management =>
void Tensor::initOnHost()
{
        floatType* rawPtr;
        cudaMallocHost(&rawPtr, m_byteSize);
        m_pHost.reset(rawPtr);
}

void Tensor::initOnDevice()
{
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&(m_pDevice), m_byteSize));
}

void Tensor::initOnDevice(cudaStream_t stream)
{
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&(m_pDevice), m_byteSize, stream));
}

void Tensor::freeMemory()
{
        m_pHost.reset();
        if (m_pDevice)
        {
                cudaFree(m_pDevice);
                m_pDevice = nullptr;
        }
}
void Tensor::cpyToDevice() const
{
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(m_pDevice, m_pHost.get(), m_byteSize, cudaMemcpyHostToDevice));
        // copy tensor to GPU
}
void Tensor::cpyToDevice(cudaStream_t stream) const
{
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(m_pDevice, m_pHost.get(), m_byteSize, cudaMemcpyHostToDevice, stream));
        // copy tensor to GPU
}

void Tensor::cpyToHost() const
{
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(m_pHost.get(), m_pDevice, m_byteSize, cudaMemcpyDeviceToHost));
        // copy tensor to Host
}
void Tensor::cpyToHost(cudaStream_t stream) const
{
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(m_pHost.get(), m_pDevice, m_byteSize, cudaMemcpyDeviceToHost, stream));
        // copy tensor to Host
}

// Getters =>
const std::vector<Index>& Tensor::getInds() const
{
        return this->m_indices;
}

const std::vector<int32_t>& Tensor::getModes() const
{
        return this->m_modes;
}
const std::vector<int64_t>& Tensor::getExtents() const
{
        return this->m_extents;
}
int Tensor::getOrder() const
{
        return this->m_order;
}
size_t Tensor::getElements() const
{
        return this->m_elements;
}
size_t Tensor::getByteSize() const
{
        return this->m_byteSize;
}

floatType* Tensor::getHostPtr() const
{
        return this->m_pHost.get();
} // WARN: returns float*

cutensorTensorDescriptor_t& Tensor::getDesc()
{
        return this->m_desc;
}
void* Tensor::getDevicePtr() const
{
        return this->m_pDevice;
}

// Set values of the Tensor =>
void Tensor::setInt(const int val, cudaStream_t stream)
{
        for (size_t j = 0; j < m_elements; ++j) // populate the tensor
        {
                m_pHost.get()[j] = val;
        }
        this->cpyToDevice(stream);
}

void Tensor::setZero(cudaStream_t stream)
{
        Tensor::setInt(0, stream);
}

void Tensor::setOne(cudaStream_t stream)
{
        Tensor::setInt(1, stream);
}

void Tensor::setRand(cudaStream_t stream)
{
        for (size_t j = 0; j < m_elements; ++j) // populate the tensor
        {
                m_pHost.get()[j] = ((floatType)rand()) / RAND_MAX;
        }
        this->cpyToDevice(stream);
}

void Tensor::setInds(const std::vector<Index>& newInds)
{
        m_indices.resize(newInds.size());
        for (int i = 0; i < newInds.size(); ++i)
        {
                m_indices[i] = newInds[i];
        }
}
// Basic unary Operations [call A.operation();] =>
floatType Tensor::fNorm() const
{
        return std::sqrt(this->fNormSquared());
}
floatType Tensor::fNormSquared() const
{
        floatType res{0};
        for (int i = 0; i < m_elements; ++i)
        {
                res += m_pHost.get()[i] * m_pHost.get()[i];
        }
        return res;
}
void Tensor::primeAll()
{
        for (int i = 0; i < m_order; ++i)
        {
                m_indices[i].prime(m_modes[i]);
        }
}

void Tensor::nextPermute() {}

void Tensor::matchPermute(const Tensor& other) {}

// Arithmetic =>

// I/O =>
static void printLevel(std::ostream& os, const floatType* data, const std::vector<int64_t>& extents,
                       const std::vector<size_t>& strides, int dim, size_t offset)
{
        int64_t N = extents[dim];

        os << "[";
        for (int64_t i = 0; i < N; ++i)
        {
                size_t idx = offset + i * strides[dim];

                if (dim + 1 == (int)extents.size())
                {
                        // Last dimension: print value
                        os << data[idx];
                }
                else
                {
                        // For 2D and higher, add newline and indent only for inner arrays (not the first)
                        if (i > 0)
                                os << "\n ";

                        printLevel(os, data, extents, strides, dim + 1, idx);
                }

                if (i + 1 < N)
                        os << ", ";
        }
        os << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& T)
{
        const auto& extents = T.getExtents();
        int order = T.getOrder();

        if (T.getElements() == 0)
        {
                os << "[]";
                return os;
        }
        if (order == 0)
        {
                // Scalar (0D tensor with 1 element)
                os << *(T.getHostPtr());
                return os;
        }
        // precompute column-major strides:
        //   stride[0] = 1
        //   stride[i] = product(extents[0..i-1])
        std::vector<size_t> strides(order, 1);
        for (int i = 1; i < order; ++i)
                strides[i] = strides[i - 1] * size_t(extents[i - 1]);

        // start recursion at dim 0, offset 0
        printLevel(os, T.getHostPtr(), extents, strides, 0, 0);
        return os;
}

// Tensor Operations =>

// Helper function to figure out the indices of C = A * B,
// returns a pair of vectors (modesC, extentsC) that we assign to (C.m_modes,C.m_extents) =>

std::pair<std::vector<int32_t>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B)
{
        // Builds a map of (B.modes, B.extents), (necessary to enforce the order A, B)
        std::unordered_map<int32_t, int64_t> mapB;

        // Since getUniqueIndsAB is not a member function we extract what we need at
        // the start
        std::vector<int32_t> modesA = A.getModes();
        std::vector<int32_t> modesB = B.getModes();
        std::vector<int64_t> extentsA = A.getExtents();
        std::vector<int64_t> extentsB = B.getExtents();

        for (size_t i = 0; i < modesB.size(); ++i)
        {
                mapB.emplace(modesB[i], extentsB[i]);
        }

        std::vector<std::pair<int32_t, int64_t>> tmp;
        tmp.reserve(modesA.size() + modesB.size());

        /*
         * Note that this is a vector of pairs, purely because it makes more sense logically to find the data
	 * (modes_i, extents_i) for a particular unique Index_i than to find all modes, then all extents in
	 * necessarily the same order.
         */

        // Adds those in A, not in B
        for (size_t j = 0; j < modesA.size(); ++j)
        {
                int mode = modesA[j];
                auto it = mapB.find(mode);
                if (it == mapB.end())
                {
                        tmp.emplace_back(mode, extentsA[j]); // Builds the pair in place
                }
                else
                {
                        // common IDs get erased
                        mapB.erase(it);
                }
        }

        // Adds those in B, not in A (common IDs have already been erased)
        for (auto& p : mapB)
        {
                tmp.emplace_back(p.first, p.second);
        }

        // Since for us it is more useful to have a pair of two separately contiguous
        // vectors in the Tensor object, we eventually return a pair<vector<size_t>,
        // vector<int64_t>>, i.e (C.m_modes, C.m_extents)
        std::vector<int32_t> modesC;
        std::vector<int64_t> extentsC;
        modesC.reserve(tmp.size());
        extentsC.reserve(tmp.size());
        for (auto& pr : tmp)
        {
                modesC.push_back(pr.first);
                extentsC.push_back(pr.second);
        }

        return {std::move(modesC), std::move(extentsC)};
}

/*
* We describe the operation to be done on input tensors; create a plan preference, estimate workspace size
* needed over the course of the operation, then create a proper plan that has information about all tunable
* parameters.
* The Operation Descriptor below encodes the contraction,
*		D_modesD = alpha * opA(A_modesA) * opB(B_modesB) + beta * opC(C_modesC)
*
* For our two-tensor contraction, we set beta to 0, and reuse our Tensor Descriptor for C, as the output Tensor D.
* Note that modes_D ≡ modes_C ≡ modes_(A * B). opA() does an operation on all elements of A, we currently use the
* identity operation.
*/

Tensor contractAB(Tensor& A, Tensor& B)
{
        CutensorHandle handle; // Would eventually be an acquired handle from a HandlePool
        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(A.getDesc()), A.getOrder(),
                                                             A.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));

        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(B.getDesc()), B.getOrder(),
                                                             B.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));
        CudaStream stream;
        if (A.getHostPtr() && A.getDevicePtr())
        {
                A.cpyToDevice(stream);
        }
        else
        {
                // TODO: Manage the error
                throw std::runtime_error("Tensor A has invalid memory pointers!");
        }
        if (B.getHostPtr() && B.getDevicePtr())
        {
                B.cpyToDevice(stream);
        }
        else
        {
                throw std::runtime_error("Tensor B has invalid memory pointers!");
        }
        auto [modesC, extentsC] = getUniqueIndsAB(A, B);
        // C only needs its (IDs, dims) for our purposes, so this initialization will do
        Tensor C(modesC, extentsC);

        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(C.getDesc()), C.getOrder(),
                                                             C.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));

        cutensorOperationDescriptor_t descOp; // will encode the operation, init. by function below
        cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F; // Precision of contraction
        HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(handle, &descOp, A.getDesc(), A.getModes().data(),
                                                        CUTENSOR_OP_IDENTITY, // descA, A.m_modes, opA
                                                        B.getDesc(), B.getModes().data(), CUTENSOR_OP_IDENTITY,
                                                        C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY,
                                                        C.getDesc(),
                                                        C.getModes().data(), // Output to D
                                                        descCompute));

        cutensorPlan_t plan = makeContractionPlan(descOp, handle);
        uint64_t workspaceSize{0}; // attempt to optimize memory allocated
        HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceSize,
                                                       sizeof(workspaceSize)));
        void* work = nullptr;
        if (workspaceSize > 0)
        {
                HANDLE_CUDA_ERROR(cudaMalloc(&work, workspaceSize));
                assert(uintptr_t(work) % kAlignment == 0); // workspace must be aligned to 128 byte-boundary
        }

        floatTypeCompute alpha{1.0f}, beta{0.0f};
        HANDLE_CUTENSOR_ERROR(cutensorContract(handle, plan, (void*)&alpha, A.getDevicePtr(), B.getDevicePtr(),
                                               (void*)&beta, C.getDevicePtr(), C.getDevicePtr(), work, workspaceSize,
                                               stream));

        // Can add a CPU routine before the sync which forces the CPU to wait for the
        // GPU to finish work
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        C.cpyToHost(stream);
        // note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr
        // into our host ptr DONE:  TODO: Free memory!
        if (work)
        {
                HANDLE_CUDA_ERROR(cudaFree(work));
        }

        HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
        HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(descOp));

        return C;
}
Tensor axpyABC(Tensor& A, Tensor& B, Tensor& C)
{
        return axpyABC(1.0f, A, B, 1.0f, C);
}

Tensor axpyABC(floatType alpha, Tensor& A, Tensor& B, floatType beta, Tensor& C)
{
        CutensorHandle handle;
        CudaStream stream;
        // Initialize D; note that D has the same modes, extents as D
        Tensor D(C.getModes(), C.getExtents());
        D.setZero(stream);
        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(A.getDesc()), A.getOrder(),
                                                             A.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));

        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(B.getDesc()), B.getOrder(),
                                                             B.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));

        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(C.getDesc()), C.getOrder(),
                                                             C.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));

        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(D.getDesc()), D.getOrder(),
                                                             D.getExtents().data(),
                                                             NULL,           // Stride (refer below! line[169])
                                                             CUTENSOR_R_32F, // Datatype: 32-bit Real Floats
                                                             kAlignment));

        if (A.getHostPtr() && A.getDevicePtr())
        {
                A.cpyToDevice(stream);
        }
        else
        {
                // TODO: Manage the error
                throw std::runtime_error("Tensor A has invalid memory pointers!");
        }
        if (B.getHostPtr() && B.getDevicePtr())
        {
                B.cpyToDevice(stream);
        }
        else
        {
                throw std::runtime_error("Tensor B has invalid memory pointers!");
        }

        if (C.getHostPtr() && C.getDevicePtr())
        {
                C.cpyToDevice(stream);
        }
        else
        {
                // TODO: Manage the error
                throw std::runtime_error("Tensor A has invalid memory pointers!");
        }
        if (D.getHostPtr() && D.getDevicePtr())
        {
                D.cpyToDevice(stream);
        }
        else
        {
                throw std::runtime_error("Tensor B has invalid memory pointers!");
        }

        cutensorOperationDescriptor_t descOp;
        cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
        HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(
            handle, &descOp, A.getDesc(), A.getModes().data(), CUTENSOR_OP_IDENTITY, B.getDesc(), B.getModes().data(),
            CUTENSOR_OP_IDENTITY, C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY, D.getDesc(),
            D.getModes().data(), // Output to D
            descCompute));

        cutensorPlan_t plan = makeContractionPlan(descOp, handle);
        uint64_t workspaceSize{0}; // attempt to optimize memory allocated
        HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceSize,
                                                       sizeof(workspaceSize)));
        void* work = nullptr;
        if (workspaceSize > 0)
        {
                HANDLE_CUDA_ERROR(cudaMalloc(&work, workspaceSize));
                assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
        }
        HANDLE_CUTENSOR_ERROR(cutensorContract(handle, plan, (void*)&alpha, A.getDevicePtr(), B.getDevicePtr(),
                                               (void*)&beta, C.getDevicePtr(), D.getDevicePtr(), work, workspaceSize,
                                               stream));

        // Can add a CPU routine before the sync which forces the CPU to wait for the
        // GPU to finish work
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        D.cpyToHost(stream);
        // note the swap from the earlier cudaMemCpy! We are copying the GPU C ptr
        // into our host ptr DONE:  TODO: Free memory!
        if (work != nullptr)
        {
                HANDLE_CUDA_ERROR(cudaFree(work));
        }
        HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
        HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
        HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(descOp));
        return D;
}

// cuSOLVER SVD =>

/*
 * A comment about stride:
 * Let's say we have a tensor A with the indices i, j, k => dim(i) = 3, dim(j) = 4, dim(k) = 5; 
 * so the order N = 3 We lay these out as, (in column-major form) 
 * [0, 0, 0] 
 * [1, 0, 0] => the first index i, updates the fastest! 
 * [2, 0, 0] 
 * [0, 1, 0] => the second index j updates every 3 (dim(i)) entries; 
 * [1, 1, 0]    We can extrapolate that k will update after every 3 * 4 = 20 (dim(i) * dim(j)) entries. 
 *     .	And in general the l-th index will update after every dim(1) * dim(2) ... dim(l-1) entries.
 *     .
 *     .
 * [2, 3, 4]
 *
 * To reshape these Tensors into matrices, say A, we group some indices to be the row_indices, and some to be 
 * the column_indices; let's say we group (j, k), and let i be separate. i still updates as before, with a 
 * 'stride' of 1. Since we need to reshape (j, k) into one unified index, we see that it will update with 
 * the old stride of the left-most index inside it, j, which means (j, k) now has stride 3. 
 * This means the vector of strides, which was {1, 3, 12} before, will now change to {1, 3}. 
 * Note that our extents and orders will also change as we need to initialize two new indices (in this case 
 * `i` can remain as is). Our extents will go from {3, 4, 5} to {3, 20}; note that the "volume" remains unchanged. 
 * To reshape our tensor, we need to simply change the strides, extents and the order in 
 * cutensorCreateDescriptor_t, and change the m_desc.
 */

// We now define the parameter split, as the integer l, such that all indices upto and including the l-th index,
// becomes the new row index, and the l+1-th to the N-th index becomes the column index; for our most useful case of
// reshaping a tensor into a matrix.
std::pair<Tensor, Tensor> lSVD(Tensor& A, int split)
{
        CusolverHandle handle;
        CudaStream stream;
        HANDLE_CUSOLVER_ERROR(cusolverDnSetStream(handle, stream));

        if (A.getHostPtr() && A.getDevicePtr())
        {
                A.cpyToDevice(stream);
        }
        else
        {
                // TODO: Manage the error
                throw std::runtime_error("Tensor A has invalid memory pointers!");
        }

        int64_t m{1}, n{1}, k;
        int i = 0;
        size_t order = A.getOrder();
        const std::vector<int64_t>& extents = A.getExtents();
        if (split < order && split > 0) // if split == m_order or 0, then we have a rank-1
        {
                for (; i < split; ++i)
                {
                        m *= extents[i];
                }
                for (; i < order; ++i)
                {
                        n *= extents[i];
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

        Tensor U(std::vector<Index>{indsU, indsS});
        U.setZero(stream);
	Tensor Vd(std::vector<Index>({indsS, indsVd}));
	Vd.setZero(stream);
        size_t allocSize = sizeof(floatTypeCompute) * k;
        floatTypeCompute* SpHost;
        HANDLE_CUDA_ERROR(cudaMallocHost(&SpHost, allocSize));
        void* SpDevice;
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&SpDevice, allocSize, stream));

        int info = 0;
        int* devInfo{nullptr};
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&devInfo, sizeof(int), stream));
        int lwork{0};

        gesvdjInfo_t gesvdj_params = NULL;
        const double tol = 1.e-6;
        const int max_sweeps = 15;
        HANDLE_CUSOLVER_ERROR(cusolverDnCreateGesvdjInfo(&gesvdj_params));
        HANDLE_CUSOLVER_ERROR(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
        HANDLE_CUSOLVER_ERROR(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

        HANDLE_CUSOLVER_ERROR(cusolverDnSgesvdj_bufferSize(
            handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, reinterpret_cast<floatTypeCompute*>(A.getDevicePtr()), m,
            reinterpret_cast<floatTypeCompute*>(SpDevice), reinterpret_cast<floatTypeCompute*>(U.getDevicePtr()),
            m,                                                         // U: m×k, lda=m
            reinterpret_cast<floatTypeCompute*>(Vd.getDevicePtr()), k, // Vᵀ: k×n, ldv=k
            &lwork, gesvdj_params));

        floatTypeCompute* workDevice = nullptr;
        HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&workDevice), sizeof(floatTypeCompute) * lwork));

        HANDLE_CUSOLVER_ERROR(cusolverDnSgesvdj(
            handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, reinterpret_cast<floatTypeCompute*>(A.getDevicePtr()), m,
            reinterpret_cast<floatTypeCompute*>(SpDevice), reinterpret_cast<floatTypeCompute*>(U.getDevicePtr()), m,
            reinterpret_cast<floatTypeCompute*>(Vd.getDevicePtr()), k, workDevice, lwork, devInfo, gesvdj_params));

        scaleVdOnDevice(reinterpret_cast<float*>(Vd.getDevicePtr()), reinterpret_cast<const float*>(SpDevice), k, n,
                        stream);

        Vd.cpyToHost(stream);
        HANDLE_CUDA_ERROR(
            cudaMemcpyAsync(SpHost, SpDevice, sizeof(floatTypeCompute) * k, cudaMemcpyDeviceToHost, stream));

        HANDLE_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
        U.cpyToHost(stream);

        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        if (info != 0)
        {
                throw std::runtime_error("SVD failed with info = " + std::to_string(info));
        }
        HANDLE_CUDA_ERROR(cudaFree(workDevice));
        HANDLE_CUDA_ERROR(cudaFree(devInfo));
        HANDLE_CUDA_ERROR(cudaFree(SpDevice));
        cusolverDnDestroyGesvdjInfo(gesvdj_params);
        HANDLE_CUDA_ERROR(cudaFreeHost(SpHost));

        return std::pair<Tensor, Tensor>{std::move(U), std::move(Vd)};
}
