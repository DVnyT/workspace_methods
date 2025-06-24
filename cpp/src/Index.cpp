#include "../include/Index.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <vector>
#include <iostream>

// Private Helpers =>
void Index::generateMode()
{
	m_mode = BitPack::generateMode(m_tags, m_prime, m_ctorState);
}

// Constructors =>
Index::Index()
: Index(1, 0)
{}	

Index::Index(int dim)
: m_dim(dim) 
{
	// No 2 indices created with this ctor should contract!
	m_tags = {};
	m_prime = 0;
	
	m_ctorState = 1;
	
	generateMode();		
}

Index::Index(int dim, int prime)
: m_dim(dim), m_prime(prime)
{
	m_tags = {};

	m_ctorState = 1;

	generateMode();
}


Index::Index(const Index& other)
: m_dim(other.m_dim), m_prime(other.m_prime), m_tags(other.m_tags), m_mode(other.m_mode)
{
	m_ctorState = 2;
	// Simply copies the mode of other without calling generateMode().
	// A new mode need not be created until this instance is modified.
	// Setters check for this flag to determine whether to call generateMode or not,
	// and subsequently set the m_ctorState to 1 (since this is now an different instance)
}

Index& Index::operator=(const Index& other) 
{
	m_dim = other.m_dim; 
	m_prime = other.m_prime; 
	m_tags = other.m_tags; 
	m_mode = other.m_mode;
	m_ctorState = 2;
	return *this;
}

// Generic Operators =>

bool Index::operator<=>(const Index& other)
{
	return std::tie(this->m_dim, this->m_mode) < std::tie(other.m_dim, other.m_mode);	
}	

std::ostream& operator<<(std::ostream& os, Index const& idx)
{

	std::cout << "Dim = " << idx.m_dim << "\n"
		  << "Prime = " << idx.m_prime << "\n"
		  << "State = " << idx.m_ctorState << "\n"
		  << "Mode = " << idx.m_mode << std::endl;
	for(int i = 0; i < 4; ++i)
	{
		for(int j = 0; j < 8; ++j)
		{
			std::cout << idx.m_tags[i][j] << " ";
		}
     		std::cout << "\n";
	}
	return os;
}

// Getters =>
int Index::getDim() const {return this->m_dim;}
std::array<std::array<char, 8>, 4> Index::getTags() const {return this->m_tags;}
int Index::getPrime() const {return this->m_prime;}
int Index::getCtorState() const {return this->m_ctorState;}
int64_t Index::getMode() const {return this->m_mode;}

// Setters => 

void Index::setDim(int newDim)
{
	m_dim = newDim;
}

void Index::prime(int primeIncrement)
{
	m_prime += primeIncrement;
}

void Index::prime()
{
    	m_prime++;  // Default increment of 1
}

// Destructors =>
Index::~Index()
{
}

