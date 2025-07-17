#pragma once

#include "Index.hpp"
#include "Tensor.hpp"

class LocalOperator
{
private:
	Index m_leftIndex;
	Index m_rightIndex;
	Index m_botIndex;
	Index m_topIndex;
	Tensor m_operator;
};
