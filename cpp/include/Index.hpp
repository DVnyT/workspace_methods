#pragma once

#include <algorithm>       // For std::min
#include <array>           // For std::array
#include <cstdint>         // For int64_t
#include <initializer_list>
#include <iostream>        // For std::ostream
#include <string>          // For std::string
#include <string_view>     // For std::string_view
#include <ranges>          // For std::ranges::range
#include <concepts>        // For concepts
#include <compare>	   // For comparison operations <=>		

// Compiled with C++20 and MSVC 

#include "BitPack.hpp"

/*
 * High-level Overview for the Index class =>
 * 1. Stores the dimension and prime level of the Index.
 * 
 * 2. Also attaches an ID to an Index called a 'mode' that persists through a move/copy. 
 *    2 indices will be contracted by cuTENSOR IF AND ONLY IF they have the same mode and dimension.
 *    cuTENSOR will only check for these 2 integers to decide whether to contract 2 indices.
 *
 * 3. Index a(3, 1); 
 *    Index b(3, 1); 
 *    // a and b will have different modes in this case and will not contract;
 *    // This is because we don't expect two indices with the same dimension to represent the same thing.
 * 
 * 4. The usual way to 'share' an Index between two Tensors A and B would simply be to write 
 *    A.m_indices[i] = a;
 *    B.m_indices[j] = a;
 *    This ensures that the i-th Index of A contracts with the j-th Index of B (off-by-one due to 0-ordering).
 *
 * 5. Setters check for whether the Index they are modifying is a copy. If it is, and it is modified, 
 *    the modified index will cease to compare equal to the first Index and will be assigned a new mode.
 *
 * 6. Copy/Move (UNMODIFIED copies contract, MODIFIED copies don't) =>
 *    Index c(a); c and a WILL contract, c is a deep copy of a.
 *    c.addPrime(2); c will NOT contract with a anymore as it has now been modified [a'' != a].
 */

class Index 
{
private:
	// Dimension. If i is the Index for a 3D space, then it will run over x, y, z for example.
	int64_t m_extent{1};
	
	// Prime the Index. So the Index j' would have prime = 1; k''' would have prime = 3.
	// 8 bits of priming (256 levels) allowed for now.
	int m_prime{0};		

	// The constructor that was called for the creation of this Index. 
	// Necessary for the bookkeeping of our copy/move semantics defined above.
	int m_ctorState{0};

	// An 32-bit ID is attached to an Index that persists through copies/moves.
	// This will end up being the Modes for our Tensors. Since dims are checked by cuTENSOR separately,
	// we bitpack the rest of the information about our Index into this 32-bit integer 
	// => 8 bits for m_prime + 4 bits for m_ctorState + 20 bits as a counter.
	void generateMode();
	int64_t m_mode{0};

public:
	// Constructors =>
	Index();
	Index(int64_t extent);		
	Index(int64_t extent, int prime);
	
        // Constructor for debugging only =>
	Index(int64_t extent, int prime, int ctorState, int64_t mode);
	
	// Copy/Move =>
	Index(const Index& other);
	Index& operator=(const Index& other);		
						
	Index(Index&& other) = default;
	Index& operator=(Index&& other) = default; 

	// Generic Operators =>
	bool operator==(const Index& other) const;
	bool operator!=(const Index& other) const;
	friend std::ostream& operator<<(std::ostream& os, Index const& idx);
	
	// Getters =>
	int64_t getExtent() const;
	int getPrime() const;

	int getCtorState() const;
	int64_t getMode() const;

	// Setters =>
	void setExtent(int newExtent);
	
	void prime();
	void prime(int increment);
	void unprime();
	void unprime(int decrement);

	// Destructors =>
	~Index();
};

