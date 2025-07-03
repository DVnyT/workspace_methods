#pragma once
#include "CudaUtils.hpp"

extern int leastPriority, greatestPriority;

struct CutensorHandle
{
        cutensorHandle_t m_handle;

        // Default ctor and dtor
        CutensorHandle();
        ~CutensorHandle();

        // No copies allowed
        CutensorHandle(const CutensorHandle &) = delete;
        CutensorHandle &operator=(const CutensorHandle &) = delete;

        // Allow moves
        CutensorHandle(CutensorHandle &&other) noexcept;
        CutensorHandle &operator=(CutensorHandle &&other) noexcept;

        // Implicit conversion to raw handle when passed.
        // CutensorHandle hand;
        // func(hand); works the same as func(hand.m_handle);
        operator cutensorHandle_t() const noexcept { return m_handle; }
};

struct CusolverHandle
{
        cusolverDnHandle_t m_handle;
        CusolverHandle();
        ~CusolverHandle();

        CusolverHandle(const CusolverHandle &) = delete;
        CusolverHandle &operator=(const CusolverHandle &) = delete;

        CusolverHandle(CusolverHandle &&other) noexcept;
        CusolverHandle &operator=(CusolverHandle &&other) noexcept;

        operator cusolverDnHandle_t() const noexcept { return m_handle; }
};

struct CublasHandle
{
        cublasHandle_t m_handle;
        CublasHandle();
        ~CublasHandle();

        CublasHandle(const CublasHandle &) = delete;
        CublasHandle &operator=(const CublasHandle &) = delete;

        CublasHandle(CublasHandle &&other) noexcept;
        CublasHandle &operator=(CublasHandle &&other) noexcept;

        operator cublasHandle_t() const noexcept { return m_handle; }
};

class HandlePool
{

};

struct CudaStream
{
        cudaStream_t m_stream;
        int m_priority;
        CudaStream();
        ~CudaStream();
        operator cudaStream_t() const noexcept { return m_stream; }
};

class StreamPool
{
        StreamPool();
};

class MemoryPool
{

};
