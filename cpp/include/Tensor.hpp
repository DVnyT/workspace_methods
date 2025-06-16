#pragma once

#include "../include/Index.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"

extern cutensorHandle_t globalHandle;
extern const uint32_t kAlignment;

class Tensor 
{
private:	
	
	// Main Objects =>
	std::vector<Index> m_indices;					// set of indices
	std::vector<int> m_modes;					// an ordered set of Index identifiers
	std::vector<int64_t> m_extents;					// dimensions of those indices
	
	// Derived Objects =>
	int m_order{0};							// number of indices
	size_t m_elements{1};						// total coefficients of the tensor	
 	size_t m_byteSize{0};						// total number of bytes to store m_elements
	
	// Tensor Coefficients =>
	// pointer to the values/coeffs. of the tensor (host); the values of the tensor are accessed with m_pHost[i]
	float* m_pHost{nullptr};	
	
	// cuTENSOR Objects =>
	cutensorTensorDescriptor_t m_desc;
	void* m_pDevice{nullptr};					// pointer to the tensor on the GPU

public:

	// Constructors =>
	Tensor();							// Default Constructor
	Tensor(const std::vector<Index>& indices);
	
	// lookupInds[id] = dim, where (id, dim)≡ (modes, extents)
	Tensor(const std::map<int, int64_t>& lookupInds);		// We only really need the IDs and dims!
	
	Tensor(const std::vector<int>& modes, const std::vector<int64_t>& extents); // another minimal ctor

	// Destructor =>
	~Tensor();                                  			// Destructor
    	
	// Copy/Move =>
	Tensor(const Tensor& other);                			// Copy Constructor
    	Tensor& operator=(const Tensor& other);     			// Copy Assignment Operator
    	Tensor(Tensor&& other) noexcept;            			// Move Constructor
    	Tensor& operator=(Tensor&& other) noexcept;

	// Memory Allocation Initializations =>
	void initOnHost();
	void initOnDevice();
	void freeMemory();
	void cpyToDevice();

	// Getters =>
	const std::vector<Index>& Tensor::getInds() const;
	const std::vector<int>& getModes() const;
    	const std::vector<int64_t>& getExtents() const;
    	size_t getOrder() const;
    	
	size_t getElements() const;
    	size_t getByteSize() const;
    	float* getHostPtr() const;

	cutensorTensorDescriptor_t getDesc() const;
	void* getDevicePtr() const;	
	
	// Give the tensor its values, (there are m_elements number of values) =>
	void setZero();						// sets all values of tensor (m_pHost[i]) to 0
	void setOne();						// set all values to 1
	void setInt(const int val);				// sets all values to val
	void setRand();						// set all values randomly (0 to 1)

	// Reshape the tensor =>
	void reshape(int split);
};
// Contracts A and B over the indices toContract, (or equivalently over the modes+extents passed) 
// If toContract is not passed, contracts over all common indices
Tensor contractAB(const Tensor& A, const Tensor& B);
Tensor contractAB(const Tensor& A, const Tensor& B, 
		  const std::vector<Index>& toContract);
Tensor contractAB(const Tensor& A, const Tensor& B,
		  const std::pair<std::vector<int>, std::vector<int64_t>>& toContract);

// Finds the indices of C = contractAB(A, B);
std::pair<std::vector<int>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B);	

