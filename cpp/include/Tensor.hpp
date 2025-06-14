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
 	size_t m_byteSize{0};						// total number of bytes to store m_element


	float* m_pHost{nullptr};					// pointer to the host (you!)
	void* m_pDevice{nullptr};					// pointer to the device (GPU)

	public:
	Tensor() = default;

	Tensor(const std::map<uintptr_t, int64_t>& lookup)
	{}
	
	Tensor(const std::vector<uintptr_t>& modes, const std::vector<int64_t>& extents)
	: m_modes(modes), m_extents(extents)
    	{}

	Tensor Contract (const Tensor& A, const Tensor&B, cutensorHandle_t& globalHandle)
	{}
};
