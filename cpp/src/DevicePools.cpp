#include "../include/DevicePools.hpp"

int leastPriority = 0;
int greatestPriority = 0;

// Default ctor and dtor
CutensorHandle::CutensorHandle()
{
        HANDLE_CUTENSOR_ERROR(cutensorCreate(&m_handle));
}
CutensorHandle::~CutensorHandle()
{
        HANDLE_CUTENSOR_ERROR(cutensorDestroy(m_handle));
}

// Allow moves
CutensorHandle::CutensorHandle(CutensorHandle &&other) noexcept : m_handle(other.m_handle)
{
        other.m_handle = static_cast<decltype(m_handle)>(nullptr);
}
CutensorHandle& CutensorHandle::operator=(CutensorHandle &&other) noexcept
{
        if (this != &other)
        {
                if (m_handle)
                        HANDLE_CUTENSOR_ERROR(cutensorDestroy(m_handle));
                m_handle = other.m_handle;
                other.m_handle = static_cast<decltype(m_handle)>(nullptr);
        }
        return *this;
}

// Implicit conversion to raw handle when passed.
// CutensorHandle hand;
// func(hand); works the same as func(hand.m_handle);

CusolverHandle::CusolverHandle()
{
        HANDLE_CUSOLVER_ERROR(cusolverDnCreate(&m_handle));
}
CusolverHandle::~CusolverHandle()
{
        HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(m_handle));
}

CusolverHandle::CusolverHandle(CusolverHandle &&other) noexcept : m_handle(other.m_handle)
{
        other.m_handle = static_cast<decltype(m_handle)>(nullptr);
}
CusolverHandle& CusolverHandle::operator=(CusolverHandle &&other) noexcept
{
        if (this != &other)
        {
                if (m_handle)
                        HANDLE_CUSOLVER_ERROR(cusolverDnDestroy(m_handle));
                m_handle = other.m_handle;
                other.m_handle = static_cast<decltype(m_handle)>(nullptr);
        }
        return *this;
}


CublasHandle::CublasHandle()
{
        HANDLE_CUBLAS_ERROR(cublasCreate_v2(&m_handle));
}
CublasHandle::~CublasHandle()
{
        HANDLE_CUBLAS_ERROR(cublasDestroy_v2(m_handle));
}

CublasHandle::CublasHandle(CublasHandle &&other) noexcept : m_handle(other.m_handle)
{
        other.m_handle = static_cast<decltype(m_handle)>(nullptr);
}
CublasHandle& CublasHandle::operator=(CublasHandle &&other) noexcept
{
        if (this != &other)
        {
                if (m_handle)
                        HANDLE_CUBLAS_ERROR(cublasDestroy_v2(m_handle));
                m_handle = other.m_handle;
                other.m_handle = static_cast<decltype(m_handle)>(nullptr);
        }
        return *this;
}

CudaStream::CudaStream() { HANDLE_CUDA_ERROR(cudaStreamCreate(&m_stream)); }
CudaStream::~CudaStream() { HANDLE_CUDA_ERROR(cudaStreamDestroy(m_stream)); }

StreamPool::StreamPool() { cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority); }
