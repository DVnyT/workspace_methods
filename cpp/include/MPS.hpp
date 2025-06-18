#pragma once

#include "Index.hpp"
#include "Tensor.hpp"
#include <vector>

class MPS
{
private:	
	// Main Objects =>
	std::vector<SiteTensor> m_sites{1};			
	int64_t m_maxDims;					
	std::vector<int64_t> m_linkDims;			// Looks something like {1, 50, 2500, 2500, 50, 1}

	// Derived Objects =>
	int m_siteNumber;					// Number of sites; m_sites.size()
	std::vector<Index> m_physIndices;			// set of {m_sites.m_phys}
	
	// Constructors =>
	// Destructors =>
	// Copy/Move =>
	
	// Operations =>
	void leftNormalizeAll();
	void rightNormalizeAll();
	void mixedNormalize(int singularSite);
	void trace();
	void innerProduct();
	void transpose();
};
