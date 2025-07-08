#pragma once

#include "Index.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <ostream> // for std::ostream
#include <tuple>
#include <vector>

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverDn.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverSp.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"

extern const uint32_t kAlignment;

using floatType = float; // NOTE: changes global precision
using floatTypeCompute = float;

class Tensor
{
      private:
        // Main object =>
        std::vector<Index> m_indices; // set of indices

        // Derived objects =>
        std::vector<int32_t> m_modes;         // index modes
        std::vector<int64_t> m_extents;       // index extents
        int m_order{0};                       // tensor order
        size_t m_elements{1};                 // total coefficients
        size_t m_byteSize{sizeof(floatType)}; // total bytes

        // Data on the Host =>
        std::unique_ptr<floatType, decltype(&cudaFreeHost)> m_pHost{nullptr, cudaFreeHost}; // host coefficients

        // Data on the Device (GPU) =>
        cutensorTensorDescriptor_t m_desc;                                       // cuTENSOR descriptor
        std::unique_ptr<void, decltype(&cudaFree)> m_pDevice{nullptr, cudaFree}; // device pointer

      public:
        // Constructors =>
        Tensor() = delete;
        explicit Tensor(const std::vector<Index>& indices);
        Tensor(const std::vector<int32_t>& modes, const std::vector<int64_t>& extents);

        // Destructor
        ~Tensor();

        // Copy/Move =>
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        // Memory management =>
        // The overloads that take in a stream, can run asynchronously.
        void initOnHost();
        void initOnDevice();
        void initOnDevice(cudaStream_t stream);
        void cpyToHost() const;
        void cpyToHost(cudaStream_t stream) const;
        void cpyToDevice() const;
        void cpyToDevice(cudaStream_t stream) const;

        // Getters =>
        const std::vector<Index>& getInds() const;
        const std::vector<int32_t>& getModes() const;
        const std::vector<int64_t>& getExtents() const;
        int getOrder() const;
        size_t getElements() const;
        size_t getByteSize() const;
        floatType* getHostPtr() const;
        cutensorTensorDescriptor_t& getDesc();
        void* getDevicePtr() const;

        // Fill operations =>
        void setZero(cudaStream_t stream);
        void setOne(cudaStream_t stream);
        void setInt(int val, cudaStream_t stream);
        void setRand(cudaStream_t stream);
        void setInds(const std::vector<Index>& newInds);

        // Other modifiers =>
        void primeAll();
        void nextPermute();
        void matchPermute(const Tensor& other);

        // Norms =>
        floatType fNorm() const;
        floatType fNormSquared() const;

        // Arithmetic =>
        //  TODO: Can consider these for small kernels on the GPU for big enough tensors?
        Tensor operator+(const Tensor& other) const;
        Tensor& operator+=(const Tensor& other);
        Tensor operator-(const Tensor& other) const;
        Tensor& operator-=(const Tensor& other);
        Tensor operator+(double scalar) const;
        Tensor operator-(double scalar) const;
        Tensor operator*(double scalar) const;
        Tensor operator/(double scalar) const;
        Tensor& operator+=(double scalar);
        Tensor& operator-=(double scalar);
        Tensor& operator*=(double scalar);
        Tensor& operator/=(double scalar);
};

// I/O =>
static void printLevel(std::ostream& os, const floatType* data, const std::vector<int64_t>& extents,
                       const std::vector<size_t>& strides, int dim, size_t offset);
std::ostream& operator<<(std::ostream& os, const Tensor& T);

// Non-member operations =>
std::pair<std::vector<int32_t>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B);
Tensor contractAB(Tensor& A, Tensor& B, cutensorHandle_t handle, cudaStream_t stream);
Tensor contractAB(Tensor& A, Tensor& B, const std::vector<Index>& toContract, cutensorHandle_t handle,
                  cudaStream_t stream);
Tensor axpyABC(Tensor& A, Tensor& B, Tensor& C, cutensorHandle_t handle, cudaStream_t stream);
Tensor axpyABC(floatType alpha, Tensor& A, Tensor& B, floatType beta, Tensor& C, cutensorHandle_t handle,
               cudaStream_t stream);

// cuSOLVER SVD =>

/// Scale the rows of an m×k matrix (column-major) U_dev by the vector S_dev (length m)
void scaleUOnDevice(floatType* U_dev, const floatType* S_dev, int m, int k, cudaStream_t stream);
/// Scale the columns of a k×n matrix (column-major) Vd_dev by the vector S_dev (length k)
void scaleVdOnDevice(floatType* Vd_dev, const floatType* S_dev, int k, int n, cudaStream_t stream);

std::pair<Tensor, Tensor> lSVD(Tensor& A, int split, cusolverDnHandle_t handle, cudaStream_t stream);
std::pair<Tensor, Tensor> rSVD(Tensor& A, int split, cusolverDnHandle_t handle, cudaStream_t stream);

class SiteTensor : public Tensor
{
};
