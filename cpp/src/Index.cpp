#pragma once

#include "../include/Index.hpp"

// Private Helpers =>
void Index::registerSelf()
{
	m_registry[m_uniqueID] = this;
 	m_uniqueID = s_globalID.fetch_add(1, std::memory_order_relaxed);
}

void Index::unregisterSelf()
{
	m_registry[m_uniqueID] = nullptr;
}

// Constructors =>
Index::Index()
{
	registerSelf();
}	

Index::Index(int dim)
: m_dim(dim) 
{
	registerSelf();
}

Index::Index(int dim, const std::vector<std::string>& tags)
: m_dim(dim), m_tags(tags)
{
	registerSelf();
}

Index::Index(int dim, int prime)
: m_dim(dim), m_prime(prime)
{
	registerSelf();
}

Index::Index(int dim, const std::vector<std::string>& tags, int prime)
: m_dim(dim), m_tags(tags), m_prime(prime)
{
	registerSelf();
}

// Getters =>
int Index::getDim() const {return this->m_dim;}
const std::vector<std::string>& Index::getTags() const {return this->m_tags;}
int Index::getPrime() const {return this->m_prime;}
int Index::getUniqueID() const {return this->m_uniqueID;}

// Setters => 
void Index::prime(int primeVal)
{
	m_prime += primeVal;
}
void Index::prime()
{
	m_prime++;
}

// ID Queries =>
Index* Index::getIndex(int uniqueID)
{
	auto it = m_registry.find(uniqueID);
	
	if(it != m_registry.end())
	{
		return it->second;
	}

	return nullptr;
}

// Destructors =>
Index::~Index()
{
	unregisterSelf();
}

