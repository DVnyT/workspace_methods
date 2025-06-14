#pragma once

#include "../include/Tensor.hpp"
#include <cstdint>

Tensor::Tensor() = default;

Tensor::Tensor(const std::map<uintptr_t, int64_t>& lookup)
{
	m_rank = 0;
	for (const auto& i: lookup)
	{
		m_modes.push_back(i.first);
		m_extents.push_back(i.second);  // DONE:  TODO: Move the logic outside the .hpp 
		m_elements *= i.second;
		m_rank++;
	}
		
	m_size = m_elements * sizeof(float);
	if (m_size != 0) 
	{
		m_pHost = (float*) malloc(m_size);
		cudaMalloc((void**)& m_pDevice, m_size);
		
		for(int j = 0; j < m_elements; j++)
		{
			m_pHost[j] = ((float) rand())/RAND_MAX + 0.5;
		}

		cudaMemcpy(m_pDevice, m_pHost, m_size, cudaMemcpyHostToDevice);
	}
}
	
Tensor::Tensor(const std::vector<uintptr_t>& modes, const std::vector<int64_t>& extents)
: m_modes(modes), m_extents(extents)
{
	m_rank = modes.size();
	for (const auto& i : extents)
	{
		m_elements *= i;
	}
		
	m_size = m_elements * sizeof(float);
	if (m_size != 0) 
	{
		m_pHost = (float*) malloc(m_size);
		cudaMalloc((void**)& m_pDevice, m_size);
		
		
		for(int j = 0; j < m_elements; j++)
		{
			m_pHost[j] = ((float) rand())/RAND_MAX + 0.5;
		}

		cudaMemcpy(m_pDevice, m_pHost, m_size, cudaMemcpyHostToDevice);
	}

}

Tensor::Tensor Contract(const Tensor::Tensor& A, const Tensor::Tensor& B, cutensorHandle_t& globalHandle)
{
	cutensorTensorDescriptor_t descA;
	cutensorTensorDescriptor_t descB;
	const uint32_t kAlignment = 128;  // TODO: Do make this a global variable!
	cutensorCreateTensorDescriptor_t(globalHandle,
				  &descA,
				  A.m_rank,
				  A.m_extents.data(),
				  NULL,
				  CUTENSOR_R_32F,
				  kAlignment);

	cutensorCreateTensorDescriptor_t(globalHandle,
				  &descB,
				  B.m_rank,
				  B.m_extents.data(),
				  NULL,
				  CUTENSOR_R_32F,
				  kAlignment);
	
	cutensorOperationDescriptor_t descOp;
	cutensorCreateContraction(globalHandle,
			   &descOp,
			   descA, A.m_modes.data(), CUTENSOR_OP_IDENTITY,
			   descB, B.m_modes.data(), CUTENSOR_OP_IDENTITY,




}


