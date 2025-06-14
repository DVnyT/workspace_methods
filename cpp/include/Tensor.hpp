#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <vector>

class Tensor 
{
	std::vector<uintptr_t> m_modes;					// an ordered set of Index hashes (= addresses)
	std::vector<int64_t> m_extents;					// dimensions of those indices
	
	size_t m_order{0};						// size(m_modes)
	size_t m_elements{1};						// total coefficients of the tensor	
 	size_t m_byteSize{0};						// total number of bytes to store m_elements

	void* m_pDevice{nullptr};					// pointer to the device (GPU)

	public:

	// Constructors =>
	Tensor();							// Default Constructor
	Tensor(const std::map<uintptr_t, int64_t>& lookup);		// Constructed via (key, value) pairs
	Tensor(const std::vector<uintptr_t>& modes, const std::vector<int64_t>& extents);	

	// Destructor =>
	~Tensor();                                  			// Destructor
    	
	// Copy/Move =>
	Tensor(const Tensor& other);                			// Copy Constructor
    	Tensor& operator=(const Tensor& other);     			// Copy Assignment Operator
    	Tensor(Tensor&& other) noexcept;            			// Move Constructor
    	Tensor& operator=(Tensor&& other) noexcept;
	
	// Getters =>
	const std::vector<uintptr_t>& getModes() const;
    	const std::vector<int64_t>& getExtents() const;
    	size_t getOrder() const;
    	size_t getElements() const;
    	size_t getByteSize() const;
    	void* getDevicePtr() const;	
};

Tensor Contract(const Tensor& A, const Tensor& B, cutensorHandle_t& globalHandle);
