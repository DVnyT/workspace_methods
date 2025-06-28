
#pragma once

#include "Index.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <vector>
#include <tuple>
#include <ostream>    // for std::ostream

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverDn.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverSp.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"

extern cutensorHandle_t globalHandle;
extern cusolverDnHandle_t solverHandle;
extern const uint32_t kAlignment;

using floatType = float;                        // NOTE: changes global precision
using floatTypeCompute = float;

class Tensor 
{
public:
        // Constructors
        Tensor() = delete;
        explicit Tensor(const std::vector<Index>& indices);
        Tensor(const std::vector<int>& modes, const std::vector<int64_t>& extents);

        // Move-only
        Tensor(const Tensor& other) = delete;
        Tensor& operator=(const Tensor& other) = delete;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        // Memory management
        void initOnHost();
        void initOnDevice();
        void freeMemory();
        void cpyToHost() const;
        void cpyToDevice() const;

        // Getters
        const std::vector<Index>& getInds() const;
        const std::vector<int>& getModes() const;
        const std::vector<int64_t>& getExtents() const;
        int getOrder() const;
        size_t getElements() const;
        size_t getByteSize() const;
        floatType* getHostPtr() const;
        cutensorTensorDescriptor_t getDesc() const;
        void* getDevicePtr() const;

        // Fill operations
        void setZero();
        void setOne();
        void setInt(int val);
        void setRand();

        // Norms & permutations
        floatType fNorm() const;
        floatType fNormSquared() const;
        void primeAll();
        void nextPermute();
        void matchPermute(const Tensor& other);

        // Arithmetic
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

        // cuSOLVER SVD
        Tensor lSVD(int split);
        Tensor rSVD(int split);

        // Destructor
        ~Tensor();

        // Grant I/O access
        friend std::ostream& operator<<(std::ostream& os, const Tensor& T);

private:
        std::vector<Index> m_indices;                  // set of indices
        std::vector<int> m_modes;                       // index modes
        std::vector<int64_t> m_extents;                 // index extents
        int m_order{0};                                 // tensor order
        size_t m_elements{1};                           // total coefficients
        size_t m_byteSize{0};                           // total bytes

        std::unique_ptr<floatType[]> m_pHost{nullptr};  // host coefficients
        cutensorTensorDescriptor_t m_desc;              // cuTENSOR descriptor
        void* m_pDevice{nullptr};                       // device pointer
};

// Non-member operations
Tensor contractAB(const Tensor& A, const Tensor& B);
Tensor contractAB(const Tensor& A, const Tensor& B, const std::vector<Index>& toContract);
Tensor axpyABC(floatType alpha, const Tensor& A, const Tensor& B, floatType beta, const Tensor& C);
std::pair<std::vector<int>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B);

// I/O declaration
std::ostream& operator<<(std::ostream& os, const Tensor& T);
