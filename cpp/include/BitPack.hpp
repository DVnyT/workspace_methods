#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>

#include "xxhash32.h"

static constexpr int BITS_HASH 	  = 32;
static constexpr int BITS_PRIME   = 8;
static constexpr int BITS_STATE   = 4;
static constexpr int BITS_COUNTER = 20;

static constexpr int SHIFT_HASH    = 0;
static constexpr int SHIFT_PRIME   = SHIFT_HASH  + BITS_HASH;
static constexpr int SHIFT_STATE   = SHIFT_PRIME + BITS_PRIME;
static constexpr int SHIFT_COUNTER = SHIFT_STATE + BITS_STATE;

static constexpr uint64_t MASK_HASH    = (1ULL << BITS_HASH)    - 1;
static constexpr uint64_t MASK_PRIME   = (1ULL << BITS_PRIME)   - 1;
static constexpr uint64_t MASK_STATE   = (1ULL << BITS_STATE)   - 1;
static constexpr uint64_t MASK_COUNTER = (1ULL << BITS_COUNTER) - 1;

static constexpr uint32_t MAX_PRIME = (1ULL << BITS_PRIME) - 1;
static constexpr uint32_t MAX_STATE = (1ULL << BITS_STATE) - 1;

static std::atomic_int32_t counter{0};

namespace BitPack
{	
	static uint32_t generateHash(const std::array<std::array<char, 8>, 4>& tags)
	{
		uint32_t seed = 0;
		return XXHash32::hash(static_cast<const void*>(tags.data()), 32, seed);
	}

	static uint64_t generateMode(const std::array<std::array<char, 8>, 4>& tags, int8_t prime, int8_t ctorState)
	{
		/*
	 	* Call generateMode() to bitpack the current state of the Index:
	 	* 32-bits hash for m_tags + 8 bits for m_prime + 4 bits for m_ctorState + 20 bits as a counter
	 	*/
        	uint32_t hash = generateHash(tags);
		int32_t thisCounter = counter.fetch_add(1);
       		
		if (hash > MASK_HASH)
		{
              		throw std::out_of_range("Hash out of range.");
		}

		if (prime > MASK_PRIME)
		{
			throw std::out_of_range("Prime out of range");	
		}

		if (ctorState > MASK_STATE)
		{
	               	throw std::out_of_range("State out of range");	
		}

		if (counter > MASK_COUNTER)
		{
	               	throw std::out_of_range("Counter out of range");		
		}

		

                return ((hash & MASK_HASH) << SHIFT_HASH)
                       | ((prime & MASK_PRIME) << SHIFT_PRIME)
                       | ((ctorState & MASK_STATE) << SHIFT_STATE)
                       | ((counter & MASK_COUNTER) << SHIFT_COUNTER);
	}
}
