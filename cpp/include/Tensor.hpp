#pragma once

#include "Index.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <vector>

class Tensor 
{
	std::vector<Index> m_indices;					// set of indices
	std::vector<size_t> m_modes;					// an ordered set of Index identifiers
	std::vector<int64_t> m_extents;					// dimensions of those indices
	
	size_t m_order{0};						// number of indices
	size_t m_elements{1};						// total coefficients of the tensor	
 	size_t m_byteSize{0};						// total number of bytes to store m_elements
	
	// pointer to the values of the tensor (host); the values of the tensor are accessed with m_pHost[i]
	float* m_pHost{nullptr};				
	
	void* m_pDevice{nullptr};					// pointer to the tensor on the GPU

	public:

	// Constructors =>
	Tensor();							// Default Constructor
	Tensor(const std::vector<Index>& indices);			// We only really need the IDs and dims!
	
	Tensor(const std::map<size_t, int64_t>& lookupInds);	
	// lookupInds[ID] = dim, where (ID,dim)≡(modes, extents)
	
	Tensor(const std::vector<size_t>& modes, const std::vector<int64_t>& extents);	

	// Destructor =>
	~Tensor();                                  			// Destructor
    	
	// Copy/Move =>
	Tensor(const Tensor& other);                			// Copy Constructor
    	Tensor& operator=(const Tensor& other);     			// Copy Assignment Operator
    	Tensor(Tensor&& other) noexcept;            			// Move Constructor
    	Tensor& operator=(Tensor&& other) noexcept;
	
	// Getters =>
	const std::vector<size_t>& getModes() const;
    	const std::vector<int64_t>& getExtents() const;
    	size_t getOrder() const;
    	size_t getElements() const;
    	size_t getByteSize() const;
    	float* getHostPtr() const;
	void* getDevicePtr() const;	
	
	
	// Give the tensor its values, (there are m_elements number of values) =>
	void setZero();						// sets all values of tensor (m_pHost[i]) to 0
	void setOne();						// set all values to 1
	void setRand();						// set all values randomly (0 to 1)

	// Reshape the tensor =>
	void reshape(const std::vector<Index>& column_Indices, const std::vector<Index>& row_Indices);

	

};

Tensor contractAB(const Tensor& A, const Tensor& B, std::vector<Index>& toContract, cutensorHandle_t globalHandle);

std::vector<size_t> getUniqueIndsAB(const Tensor& A, const Tensor&B);	// Finds the indices of C = contractAB(A, B);

