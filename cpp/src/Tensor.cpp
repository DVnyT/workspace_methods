#include "../include/Tensor.hpp"
#include "../include/CudaUtils.hpp"
#include "../include/DevicePools.hpp"
#include "../include/Index.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <format>
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
constexpr int max_precision = std::numeric_limits<double>::max_digits10; // To print/log with max precision

// Constructors =>
// DONE:  TODO: Move the logic outside the .hpp

Tensor::Tensor(const std::vector<Index>& indices) : m_indices(indices), m_order(indices.size()), m_elements(1)
{
        m_modes.resize(m_order);
        m_extents.resize(m_order);
        for (size_t i = 0; i < m_order; ++i)
        {
                m_modes[i] = indices[i].getMode();
                m_extents[i] = indices[i].getExtent();
                m_elements *= m_extents[i]; // Total number of coefficients of the tensor = product(all extents)
        }
        m_byteSize = sizeof(floatType) * m_elements;

        this->initOnHost(); // Allocates pinned memory (on the Host, but addressible by the GPU.)
                            // Speeds up Host<->Device transfers.
        this->initOnDevice();
}

Tensor::Tensor(const std::vector<int32_t>& modes, const std::vector<int64_t>& extents)
    : m_modes(modes), m_extents(extents), m_order(modes.size()), m_elements(1)
{
        for (const auto& i : extents)
        {
                m_elements *= i;
        }
        // Indices stay empty!

        m_byteSize = sizeof(floatType) * m_elements;

        this->initOnHost();
        this->initOnDevice();
}

Tensor::~Tensor()
{
        // Pointer deletion is handled by unique pointers.
}
// Copy/Move =>

// Memory Management =>
void Tensor::initOnHost()
{
        if (m_byteSize)
        {
                floatType* rawPtr{nullptr};
                HANDLE_CUDA_ERROR(cudaMallocHost(&rawPtr, m_byteSize)); // Pinned memory.
                if (rawPtr)
                {
                        m_pHost.reset(rawPtr);
                }
                else
                {
                        std::cout << "Malloc did not go through!";
                }
        }
}
void Tensor::initOnDevice()
{
        if (m_byteSize)
        {
                void* rawPtr{nullptr};
                HANDLE_CUDA_ERROR(cudaMalloc((void**)&(rawPtr), m_byteSize));
                if (rawPtr)
                {
                        m_pDevice.reset(rawPtr);
                }
                else
                {
                        std::cout << "cudaMalloc did not go through!";
                }
        }
}

void Tensor::cpyToDevice() const
{
        if (m_pHost && m_pDevice) // Ensure allocation. Neither must be nullptr.
        {
                // copy tensor to GPU;
                HANDLE_CUDA_ERROR(cudaMemcpy(m_pDevice.get(), m_pHost.get(), m_byteSize, cudaMemcpyHostToDevice));
        }
        else
        {
                throw std::runtime_error("Memory not allocated for either m_pHost or m_pDevice!");
        }
}
void Tensor::cpyToDevice(cudaStream_t stream) const
{
        if (m_pHost && m_pDevice) // Ensure allocation. Neither must be nullptr.
        {
                // copy tensor to GPU; async with respect to other streams as well.
                HANDLE_CUDA_ERROR(
                    cudaMemcpyAsync(m_pDevice.get(), m_pHost.get(), m_byteSize, cudaMemcpyHostToDevice, stream));
        }
        else
        {
                throw std::runtime_error("Memory not allocated for either m_pHost or m_pDevice!");
        }
}

void Tensor::cpyToHost() const
{
        if (m_pHost && m_pDevice) // Ensure allocation. Neither must be nullptr.
        {
                HANDLE_CUDA_ERROR(cudaMemcpy(m_pHost.get(), m_pDevice.get(), m_byteSize, cudaMemcpyDeviceToHost));
        }
        else
        {
                throw std::runtime_error("Memory not allocated for either m_pHost or m_pDevice!");
        }
}
void Tensor::cpyToHost(cudaStream_t stream) const
{
        if (m_pHost && m_pDevice) // Ensure allocation. Neither must be nullptr.
        {
                HANDLE_CUDA_ERROR(
                    cudaMemcpyAsync(m_pHost.get(), m_pDevice.get(), m_byteSize, cudaMemcpyDeviceToHost, stream));
        }
        else
        {
                throw std::runtime_error("Memory not allocated for either m_pHost or m_pDevice!");
        }
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
floatType* Tensor::getHostPtr() const // WARN: returns the raw pointer!
{
        return this->m_pHost.get();
}
cutensorTensorDescriptor_t& Tensor::getDesc()
{
        return this->m_desc;
}
void* Tensor::getDevicePtr() const
{
        return this->m_pDevice.get(); // WARN: returns the raw pointer!
}

// Set values of the Tensor =>
void Tensor::setZero(cudaStream_t stream)
{
        Tensor::setInt(0, stream);
}

void Tensor::setOne(cudaStream_t stream)
{
        Tensor::setInt(1, stream);
}
void Tensor::setInt(const int val, cudaStream_t stream)
{
        auto tmp = this->getHostPtr();
        for (size_t j = 0; j < m_elements; ++j) // populate the tensor
        {
                tmp[j] = val;
        }
        this->cpyToDevice(stream);
}
void Tensor::setRand(cudaStream_t stream)
{
        auto tmp = this->getHostPtr();
        for (size_t j = 0; j < m_elements; ++j) // populate the tensor
        {
                tmp[j] = ((floatType)rand()) / RAND_MAX;
        }
        this->cpyToDevice(stream);
}
// For custom values, note that getHostPtr() returns the raw ptr. Column-major ordering is followed by cuTENSOR,
// cuSOLVER etc.

void Tensor::setInds(const std::vector<Index>& newInds)
{
        m_order = newInds.size();
        m_indices.resize(m_order);
        m_extents.resize(m_order);
        m_elements = 1;
        for (int i = 0; i < m_order; ++i)
        {
                m_indices[i] = newInds[i];
                m_extents[i] = newInds[i].getExtent();
                m_elements *= m_extents[i];
        }
        m_byteSize = sizeof(floatType) * m_elements;
}

// Other modifiers =>
void Tensor::primeAll()
{
        for (int i = 0; i < m_order; ++i)
        {
                m_indices[i].prime();
        }
}

void Tensor::nextPermute() {}

void Tensor::matchPermute(const Tensor& other) {}

// Norms =>
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

// Arithmetic =>

// I/O =>
static void printLevel(std::ostream& os, const floatType* data, const std::vector<int64_t>& extents,
                       const std::vector<size_t>& strides, int dim, size_t offset)
{
        int64_t N = extents[dim];
        os << "[";
        size_t idx;
        auto order = extents.size();
        for (int64_t i = 0; i < N; ++i)
        {
                idx = offset + i * strides[dim];

                if (dim + 1 == order)
                {
                        // Last dimension: print value
                        os << std::format("{:.{}f}", data[idx], max_precision);
                }
                else
                {
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
        const std::vector<int64_t>& extents{T.getExtents()};
        int order = T.getOrder();

        if (T.getElements() == 0)
        {
                os << "[]";
                return os;
        }
        if (order == 0)
        {
                os << *(T.getHostPtr());
                return os;
        }

        std::vector<size_t> strides(order, 1);
        for (int i = 1; i < order; ++i)
        {
                strides[i] = strides[i - 1] * size_t(extents[i - 1]);
        }

        printLevel(os, T.getHostPtr(), extents, strides, 0, 0);

        return os;
}

// Tensor Operations =>

// Helper function to figure out the indices of C = A * B,
// returns a pair of vectors (modesC, extentsC) that we assign to (C.m_modes,C.m_extents) =>

std::pair<std::vector<int32_t>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B)
{
        // Since getUniqueIndsAB is not a member function we extract what we need at the start
        std::vector<int32_t> modesA = A.getModes();
        std::vector<int32_t> modesB = B.getModes();
        std::vector<int64_t> extentsA = A.getExtents();
        std::vector<int64_t> extentsB = B.getExtents();

        // Builds a map of (B.modes, B.extents), (necessary to enforce the order A, B)
        std::unordered_map<int32_t, int64_t> mapB;

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

        // Adds those in A, not in B.

        for (size_t j = 0; j < modesA.size(); ++j)
        {
                auto it = mapB.find(modesA[j]);
                if (it == mapB.end())
                {
                        tmp.emplace_back(modesA[j], extentsA[j]); // Builds the pair in place
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
        for (auto& pr : tmp)
        {
                modesC.emplace_back(pr.first);
                extentsC.emplace_back(pr.second);
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

Tensor contractAB(Tensor& A, Tensor& B, cutensorHandle_t handle, cudaStream_t stream)
{
        // Creates Tensor descriptors for the input tensors
        // The NULL is when we use default strides. Strides are discussed below where we write the logic for SVDs.
        // The kAlignment = 128 is the 128-byte boundary that our descriptor must be aligned to.
        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(A.getDesc()), A.getOrder(),
                                                             A.getExtents().data(), NULL, CUTENSOR_R_32F, kAlignment));

        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(B.getDesc()), B.getOrder(),
                                                             B.getExtents().data(), NULL, CUTENSOR_R_32F, kAlignment));

        // TODO: Can let the user ensure that there are values on the Device instead of this check.
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
        // C only needs its (IDs, dims) for our purposes, so this initialization will do.
        Tensor C(modesC, extentsC);

        // We make our last tensor descriptor.
        HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(handle, &(C.getDesc()), C.getOrder(),
                                                             C.getExtents().data(), NULL, CUTENSOR_R_32F, kAlignment));

        cutensorOperationDescriptor_t descOp; // will encode the operation, initialized by function below.
        cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F; // Precision of contraction.

        HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(handle, &descOp, A.getDesc(), A.getModes().data(),
                                                        CUTENSOR_OP_IDENTITY, // descA, A.m_modes, opA
                                                        B.getDesc(), B.getModes().data(), CUTENSOR_OP_IDENTITY,
                                                        C.getDesc(), C.getModes().data(), CUTENSOR_OP_IDENTITY,
                                                        C.getDesc(), C.getModes().data(), descCompute));

        // TODO: Cache workspace sizes and reuse them!
        cutensorPlan_t plan = makeContractionPlan(descOp, handle);
        uint64_t workspaceSize{0}; // attempt to optimize memory allocated
        HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &workspaceSize,
                                                       sizeof(workspaceSize)));
        void* work = nullptr;
        if (workspaceSize > 0)
        {
                // TODO: Replace this cudaMalloc with an acquire() from a memory pool.
                HANDLE_CUDA_ERROR(cudaMalloc(&work, workspaceSize));
        }

        floatType alpha{1.0f}, beta{0.0f};

        HANDLE_CUTENSOR_ERROR(cutensorContract(handle, plan, (void*)&alpha, A.getDevicePtr(), B.getDevicePtr(),
                                               (void*)&beta, C.getDevicePtr(), C.getDevicePtr(), work, workspaceSize,
                                               stream));
        // Can add a CPU routine before the sync which forces the CPU to wait for the
        // GPU to finish work

        C.cpyToHost(stream);

        if (work)
        {
                HANDLE_CUDA_ERROR(cudaFree(work));
        }
        HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
        HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(descOp));
        return C;
}

Tensor axpyABC(Tensor& A, Tensor& B, Tensor& C, cutensorHandle_t handle, cudaStream_t stream)
{
        return axpyABC(1.0f, A, B, 1.0f, C, handle, stream);
}

Tensor axpyABC(floatType alpha, Tensor& A, Tensor& B, floatType beta, Tensor& C, cutensorHandle_t handle,
               cudaStream_t stream)
{
        // Initialize D; note that D has the same modes, extents as C.
        Tensor D(C.getModes(), C.getExtents());
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
        }
        HANDLE_CUTENSOR_ERROR(cutensorContract(handle, plan, (void*)&alpha, A.getDevicePtr(), B.getDevicePtr(),
                                               (void*)&beta, C.getDevicePtr(), D.getDevicePtr(), work, workspaceSize,
                                               stream));

        D.cpyToHost(stream);

        if (work != nullptr)
        {
                HANDLE_CUDA_ERROR(cudaFree(work));
        }
        HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
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
template <typename T> __global__ void scale_cols_kernel(T* mat, const T* vec, int rows, int cols)
{
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < rows && col < cols)
        {
                int idx = row + col * rows; // column-major
                mat[idx] *= vec[col];
        }
}
// For U matrix (m×k) - scale columns 
void scaleUOnDevice(floatType* U_dev, const floatType* S_dev, int m, int k, cudaStream_t stream)
{
        dim3 block(32, 32);
        dim3 grid((k + 31) / 32, (m + 31) / 32); // k cols, m rows
        (scale_cols_kernel<floatType>)<<<grid, block, 0, stream>>>(U_dev, S_dev, m, k);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
}

// For Vd matrix (k×n) - scale rows
void scaleVdOnDevice(floatType* Vd_dev, const floatType* S_dev, int k, int n, cudaStream_t stream)
{
        dim3 block(32, 32);
        dim3 grid((n + 31) / 32, (k + 31) / 32); // n cols, k rows
        scale_rows_kernel<floatType><<<grid, block, 0, stream>>>(Vd_dev, S_dev, k, n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
}

// TODO: Change the main handle to Cutensornet and do SVDs with it. Cutensornet handles truncation nicely.
std::pair<Tensor, Tensor> lSVD(Tensor& A, int split, cusolverDnHandle_t handle, cudaStream_t stream)
{
        // TODO: Move this stream-setting outside the function.
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

        int64_t m{1}, n{1}, k{1};
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
        Tensor Vd(std::vector<Index>{indsS, indsVd});

        size_t allocSize = sizeof(floatType) * k;
        void* SpDevice;
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&SpDevice, allocSize, stream));

        int info = 0;
        int* devInfo{nullptr};
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&devInfo, sizeof(int), stream));
        int lwork{0};

        gesvdjInfo_t gesvdj_params = NULL;
        HANDLE_CUSOLVER_ERROR(cusolverDnCreateGesvdjInfo(&gesvdj_params));

        HANDLE_CUSOLVER_ERROR(cusolverDnSgesvdj_bufferSize(
            handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, reinterpret_cast<floatType*>(A.getDevicePtr()), m,
            reinterpret_cast<floatType*>(SpDevice), reinterpret_cast<floatType*>(U.getDevicePtr()),
            m,                                                  // U: m×k, lda=m
            reinterpret_cast<floatType*>(Vd.getDevicePtr()), k, // Vᵀ: k×n, ldv=k
            &lwork, gesvdj_params));

        floatType* workDevice = nullptr;
        HANDLE_CUDA_ERROR(cudaMallocAsync(reinterpret_cast<void**>(&workDevice), sizeof(floatType) * lwork, stream));

        HANDLE_CUSOLVER_ERROR(cusolverDnSgesvdj(
            handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, reinterpret_cast<floatType*>(A.getDevicePtr()), m,
            reinterpret_cast<floatType*>(SpDevice), reinterpret_cast<floatType*>(U.getDevicePtr()), m,
            reinterpret_cast<floatType*>(Vd.getDevicePtr()), k, workDevice, lwork, devInfo, gesvdj_params));

        scaleVdOnDevice(reinterpret_cast<floatType*>(Vd.getDevicePtr()), reinterpret_cast<const floatType*>(SpDevice),
                        k, n, stream);
        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
        Vd.cpyToHost(stream);

        HANDLE_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
        U.cpyToHost(stream);

        HANDLE_CUDA_ERROR(cudaFreeAsync(workDevice, stream));
        HANDLE_CUDA_ERROR(cudaFreeAsync(devInfo, stream));
        HANDLE_CUDA_ERROR(cudaFreeAsync(SpDevice, stream));
        cusolverDnDestroyGesvdjInfo(gesvdj_params);

        return std::pair<Tensor, Tensor>{std::move(U), std::move(Vd)};
}

std::pair<Tensor, Tensor> rSVD(Tensor& A, int split, cusolverDnHandle_t handle, cudaStream_t stream)
{
        // TODO: Move this stream-setting outside the function.
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

        int64_t m{1}, n{1}, k{1};
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
        Tensor Vd(std::vector<Index>{indsS, indsVd});

        size_t allocSize = sizeof(floatType) * k;
        void* SpDevice;
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&SpDevice, allocSize, stream));

        int info = 0;
        int* devInfo{nullptr};
        HANDLE_CUDA_ERROR(cudaMallocAsync((void**)&devInfo, sizeof(int), stream));
        int lwork{0};

        gesvdjInfo_t gesvdj_params = NULL;
        HANDLE_CUSOLVER_ERROR(cusolverDnCreateGesvdjInfo(&gesvdj_params));

        HANDLE_CUSOLVER_ERROR(cusolverDnSgesvdj_bufferSize(
            handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, reinterpret_cast<floatType*>(A.getDevicePtr()), m,
            reinterpret_cast<floatType*>(SpDevice), reinterpret_cast<floatType*>(U.getDevicePtr()),
            m,                                                  // U: m×k, lda=m
            reinterpret_cast<floatType*>(Vd.getDevicePtr()), k, // Vᵀ: k×n, ldv=k
            &lwork, gesvdj_params));

        floatType* workDevice = nullptr;
        HANDLE_CUDA_ERROR(cudaMallocAsync(reinterpret_cast<void**>(&workDevice), sizeof(floatType) * lwork, stream));

        HANDLE_CUSOLVER_ERROR(cusolverDnSgesvdj(
            handle, CUSOLVER_EIG_MODE_VECTOR, 1, m, n, reinterpret_cast<floatType*>(A.getDevicePtr()), m,
            reinterpret_cast<floatType*>(SpDevice), reinterpret_cast<floatType*>(U.getDevicePtr()), m,
            reinterpret_cast<floatType*>(Vd.getDevicePtr()), k, workDevice, lwork, devInfo, gesvdj_params));

        scaleUOnDevice(reinterpret_cast<floatType*>(U.getDevicePtr()), reinterpret_cast<const floatType*>(SpDevice), m,
                       k, stream);

        HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

        Vd.cpyToHost(stream);
        HANDLE_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
        U.cpyToHost(stream);

        HANDLE_CUDA_ERROR(cudaFreeAsync(workDevice, stream));
        HANDLE_CUDA_ERROR(cudaFreeAsync(devInfo, stream));
        HANDLE_CUDA_ERROR(cudaFreeAsync(SpDevice, stream));
        cusolverDnDestroyGesvdjInfo(gesvdj_params);

        return std::pair<Tensor, Tensor>{std::move(U), std::move(Vd)};
}
