#pragma once
#include "Tensor.hpp"

class SiteTensor : public Tensor
{
	int m_siteNum{0};
	SiteIndex m_leftIndex;
	SiteIndex m_physIndex;
	SiteIndex m_rightIndex;
	
	// TODO: Symmetries.
	
};
