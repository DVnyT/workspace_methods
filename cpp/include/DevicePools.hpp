#pragma once
#include "CudaUtils.hpp"
#include <array>
#include <deque>
#include <future>
#include <mutex>
#include <vector>

extern int leastPriority, greatestPriority;
constexpr size_t handleSize = 4;


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
	static std::deque<CudaStream> m_streamQ;
	
	std::array<CutensorHandle, handleSize> m_cutensorHandlePool;
	std::array<CusolverHandle, handleSize> m_cusolverHandlePool;
	std::array<CublasHandle, handleSize> m_cublasHandlePool;
	
	std::mutex m_mtx;
	StreamPool();
	~StreamPool();
	
	CudaStream acquire();
	void putBack();
};

class MemoryPool
{
	std::array<void*, 1024> m_pinnedMemPool;
};
