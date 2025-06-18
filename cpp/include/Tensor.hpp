#pragma once

#include "Index.hpp"
#include <cstdlib>
#include <map>
#include <memory>

#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include/cutensor.h"

extern cutensorHandle_t globalHandle;
extern const uint32_t kAlignment;

using floatType = float;						// changes global precision
	
class Tensor 
{
private:	
	// Main Objects (you only need either m_indices OR m_modes [see Index.hpp to know why]) =>
	std::vector<Index> m_indices;					// set of indices
	std::vector<int> m_modes;					// an ordered set of Index m_uniqueIDs
	
	// Derived Objects =>
	std::vector<int64_t> m_extents;					// dimensions of those indices
	int m_order{0};							// number of indices
	size_t m_elements{1};						// total coefficients of the tensor	
 	size_t m_byteSize{0};						// total number of bytes to store m_elements
	
	// Tensor Coefficients =>
	// pointer to the values/coeffs. of the tensor (host); the values of the tensor are accessed with m_pHost[i]
	std::unique_ptr<floatType[]> m_pHost{nullptr};	
	
	// cuTENSOR Objects =>
	cutensorTensorDescriptor_t m_desc;				// A tensor descriptor for cuTENSOR
	void* m_pDevice{nullptr};					// pointer to the tensor on the GPU

public:

	// Constructors =>
	Tensor();							// Default Constructor
	Tensor(const std::vector<Index>& indices);
	
	// Constructors that we call internally (users won't generally know the the uniqueIDs for modes) => 
	Tensor(const std::vector<int>& modes, const std::vector<int64_t>& extents); 

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

    	floatType* getHostPtr() const;

	cutensorTensorDescriptor_t getDesc() const;
	void* getDevicePtr() const;	
	
	// Give the tensor its values, (there are m_elements number of values) =>
	void setZero();						// sets all values of tensor (m_pHost[i]) to 0
	void setOne();						// set all values to 1
	void setInt(const int val);				// sets all values to val
	void setRand();						// set all values randomly (0 to 1)
	
	// Operations on 1 tensor =>
	void reshape(int split);				// refer Tensor.cpp

	// Destructor =>
	~Tensor();                                  			
};

class SiteTensor: public Tensor
{
private:
	int m_siteNumber;
	Index m_left;
	Index m_phys;
	Index m_right;
};

// 2-Tensor Operations =>
// Not member functions! Chosen this way to give A and B equal footing.

// Contracts A and B over the indices toContract, (or equivalently over the modes+extents passed) 
// If toContract is not passed, contracts over all common indices;
Tensor contractAB(const Tensor& A, const Tensor& B);
Tensor contractAB(const Tensor& A, const Tensor& B, 
		  const std::vector<Index>& toContract);

// Finds the indices of C = contractAB(A, B); i.e. symmetric difference of A and B
std::pair<std::vector<int>, std::vector<int64_t>> getUniqueIndsAB(const Tensor& A, const Tensor& B);

