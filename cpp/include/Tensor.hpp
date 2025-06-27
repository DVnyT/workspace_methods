#pragma once

#include "Index.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <vector>
#include <tuple>

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverDn.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cusolverSp.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"

extern cutensorHandle_t globalHandle;
extern cusolverDnHandle_t solverHandle;
extern const uint32_t kAlignment;

using floatType = float;						// NOTE: changes global precision
using floatTypeCompute = float;

/*
 * A high-level overview for the Tensor class =>
 *
 *
 *
 * */

class Tensor 
{
private:	
	// Main object =>
	std::vector<Index> m_indices;					// set of indices
	
	// Derived Objects =>
	std::vector<int> m_modes;					// an ordered set of Index m_mode(s)
	std::vector<int64_t> m_extents;					// dimensions of those indices
	int m_order{0};							// number of indices
	size_t m_elements{1};						// total number of coefficients	
 	size_t m_byteSize{0};						// total number of bytes to store m_elements

	// Tensor Coefficients =>
	// pointer to the values/coeffs. of the tensor (host); 
	// the values of the tensor are accessed with A.m_pHost[i], or A.getHost()[i]
	std::unique_ptr<floatType[]> m_pHost{nullptr};	
	
	// cuTENSOR Objects =>
	cutensorTensorDescriptor_t m_desc;				// A tensor descriptor for cuTENSOR
	void* m_pDevice{nullptr};					// pointer to the tensor on the GPU

public:
	// Constructors =>
	Tensor() = delete;						// Default ctor would not work for us	
	Tensor(const std::vector<Index>& indices);
	
	// Constructors that we call internally => 
	Tensor(const std::vector<int>& modes, const std::vector<int64_t>& extents); 

	// Copy =>
	Tensor(const Tensor& other) = delete;				// We have a unique_ptr m_pHost
    	Tensor& operator=(const Tensor& other) = delete;

	// Move =>
	Tensor(Tensor&&) noexcept = default;
	Tensor& operator=(Tensor&&) noexcept = default;
	
	// Memory Management =>
	void initOnHost();
	void initOnDevice();
	void freeMemory();
	void cpyToHost() const;
	void cpyToDevice() const;

	// Getters =>
	const std::vector<Index>& getInds() const;
	
	const std::vector<int>& getModes() const;
    	const std::vector<int64_t>& getExtents() const;
	int getOrder() const;
	size_t getElements() const;
    	size_t getByteSize() const;

    	floatType* getHostPtr() const;				// WARN: returns m_pHost.get() which is a raw float*

	cutensorTensorDescriptor_t getDesc() const;
	void* getDevicePtr() const;	
	
	// Give the tensor its values, (there are m_elements number of values) =>
	void setZero();						// sets all values of tensor (m_pHost[i]) to 0
	void setOne();						// set all values to 1
	void setInt(const int val);				// sets all values to val
	void setRand();						// set all values randomly (0 to 1)
	
	// Basic unary Operations [call A.operation();] =>	
	floatType fNorm();						// Computes the Frobenius Norm
	floatType fNormSquared();
	void primeAll();						// Primes all indices by 1
	void nextPermute();					// Permutes the indices and (modes, extents)
	
	// Basic binary Operations [call A.operator(B) or A (operator) B] =>
	void matchPermute(const Tensor& other);		// Permutes the indices of this to match the order of other
	
	Tensor operator+(const Tensor& other);		// TODO: These operator overloads! For permuted indices? Y/N
	Tensor& operator+=(const Tensor& other);
	Tensor operator-(const Tensor& other);
	Tensor& operator-=(const Tensor& other);

	// Operations with scalars =>
	Tensor operator+(double scalar) const;
    	Tensor operator-(double scalar) const;
    	Tensor operator*(double scalar) const;  
    	Tensor operator/(double scalar) const;
    
    	Tensor& operator+=(double scalar);
    	Tensor& operator-=(double scalar);
    	Tensor& operator*=(double scalar);
    	Tensor& operator/=(double scalar);

	// Refer below for contractions and axpy!

	// Destructor =>
	~Tensor();	

private:
	/*
	* flatten -> matrix operation -> unflatten is a closed loop of member function calls that
	* 1) flattens the tensor to a matrix, splitting the tensor indices into 2 groups, the row indices
	* and the column indices
	* 2) operates on the resultant order-2 tensor by ONLY accessing its extents, strides and order; using
	* cuSOLVER and returns some tuple with the required data
	* 3) unflattens the matrix into the original tensor
	* Only part 2) of the functionality is given to the user directly, and the flattening processes are 
	* abstracted away since it would be hard to keep track of the tensor's internal state otherwise
	*/
	void flatten(int split);							// refer Tensor.cpp
	void unflatten(const std::vector<int64_t>& targetExtents, int targetOrder);	// Restores original Tensor 

public: 
	// cuSOLVER wrappers =>
	
	// a) SVD: split here will take an integer 0 < i < m_order such that indices [0, i) count as row indices, 
	//    and [i, m_order) count as column indices
	std::tuple<Tensor, Tensor, Tensor> svd(int split);
	
	// TODO: Other possible functions
};

// Basic binary Operations (contd.) =>
// Not member functions! Chosen this way to give A and B equal footing and to avoid overloads of * and other
// sensitive operators

// Contracts A and B over the indices toContract;
// If toContract is not passed, contracts over all common indices;
Tensor contractAB(const Tensor& A, const Tensor& B);
Tensor contractAB(const Tensor& A, const Tensor& B, const std::vector<Index>& toContract);

// Generic axpy but on tensors: D = alpha * A * B + beta * C
Tensor axpyABC(floatType alpha, const Tensor& A, const Tensor& B, floatType beta, const Tensor& C);

// Finds the indices of C = contractAB(A, B); i.e. symmetric difference of A and B
// Helper function for contractAB to find the shape of C ahead of time, returns the pair {modes, extents} 
std::pair<std::vector<int>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B);



