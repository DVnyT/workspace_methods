#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <vector>

class Tensor 
{
	std::vector<uintptr_t> m_modes;
	std::vector<int64_t> m_extents;	
	size_t m_rank;

	size_t m_elements{1};
	size_t m_size{0};
 
	float* m_pHost{nullptr};
	void* m_pDevice{nullptr};

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
