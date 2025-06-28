#include "../include/Index.hpp"
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <tuple>

// Private Helpers =>
void Index::generateMode()
{
	m_mode = BitPack::generateMode(m_prime, m_ctorState);
}

// Constructors =>
Index::Index()
: Index(1, 0)
{}	

Index::Index(int64_t extent)
: m_extent(extent) 
{
	// No 2 indices created with this ctor should contract!
	m_prime = 0;
	m_ctorState = 1;
	generateMode();		
}

Index::Index(int64_t extent, int prime)
: m_extent(extent), m_prime(prime)
{
	m_ctorState = 1;
	generateMode();
}

Index::Index(const Index& other)
: m_extent(other.m_extent), m_prime(other.m_prime), m_mode(other.m_mode)
{
	m_ctorState = 2;
	// Simply copies the mode of other without calling generateMode().
	// A new mode need not be created until this instance is modified.
	// Setters check for this flag to determine whether to call generateMode or not,
	// and subsequently set the m_ctorState to 1 (since this is now an different instance)
}

Index& Index::operator=(const Index& other) 
{
	if (other == *this){return *this;}
	m_extent = other.m_extent; 
	m_prime = other.m_prime; 
	m_mode = other.m_mode;
	m_ctorState = 2;
	return *this;
}

// Generic Operators =>

std::ostream& operator<<(std::ostream& os, Index const& idx)
{
	os << "Dim = " << idx.m_extent << "\n"
		  << "Prime = " << idx.m_prime << "\n"
		  << "State = " << idx.m_ctorState << "\n"
		  << "Mode (in bits) = " << std::bitset<sizeof(idx.m_mode)*8>(idx.m_mode) << '\n'
		  << "Mode (in decimals)= " << idx.m_mode << '\n';
	return os;
}

bool Index::operator==(const Index& other) const
{
	return (this->m_extent == other.m_extent) && (this-> m_mode == other.m_mode);
}

bool Index::operator!=(const Index& other) const
{
	return (this->m_extent != other.m_extent) || (this-> m_mode != other.m_mode);
}


// Getters =>
int64_t Index::getExtent() const {return this->m_extent;}
int Index::getPrime() const {return this->m_prime;}
int Index::getCtorState() const {return this->m_ctorState;}
int64_t Index::getMode() const {return this->m_mode;}

// Setters => 
void Index::setExtent(int newExtent)
{
	if(newExtent != m_extent)
	{
		m_extent = newExtent;
		if(m_ctorState == 2)
		{
			m_ctorState = 1;
			generateMode();
		}
	}
}

void Index::prime(int primeIncrement)
{
	if (primeIncrement)
	{
		m_prime += primeIncrement;
		if(m_ctorState == 2)
		{	
			m_ctorState = 1;
			generateMode();
		}
	}
}
void Index::prime()
{
    	m_prime++;  // Default increment of 1	
	if(m_ctorState == 2)
	{
		m_ctorState = 1;
		generateMode();
	}
}

void Index::unprime()
{
	prime(-1);
}

void Index::unprime(int decrement)
{
	prime(-decrement);
}

// Destructors =>
Index::~Index()
{}

