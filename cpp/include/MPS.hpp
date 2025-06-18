#pragma once

#include "Index.hpp"
#include "Tensor.hpp"
#include <vector>

class MPS
{
private:	
	std::vector<SiteTensor> m_sites;			
	std::vector<PhysIndex> m_physIndices;
	int64_t m_maxDims;
	std::vector<int64_t> m_linkDims;			// Looks something like {1, 50, 2500, 2500, 50, 1}
	
	

};
