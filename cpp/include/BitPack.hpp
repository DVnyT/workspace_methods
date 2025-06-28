#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>

static constexpr int BITS_PRIME   = 8;
static constexpr int BITS_STATE   = 4;
static constexpr int BITS_COUNTER = 20;

static constexpr int SHIFT_PRIME   = 0;
static constexpr int SHIFT_STATE   = SHIFT_PRIME + BITS_PRIME;
static constexpr int SHIFT_COUNTER = SHIFT_STATE + BITS_STATE;

static constexpr uint64_t MASK_PRIME   = (1ULL << BITS_PRIME)   - 1;
static constexpr uint64_t MASK_STATE   = (1ULL << BITS_STATE)   - 1;
static constexpr uint64_t MASK_COUNTER = (1ULL << BITS_COUNTER) - 1;

static constexpr uint32_t MAX_PRIME = (1ULL << BITS_PRIME) - 1;
static constexpr uint32_t MAX_STATE = (1ULL << BITS_STATE) - 1;

static std::atomic_int32_t counter{0};

namespace BitPack
{	
	static uint64_t generateMode(int8_t prime, int8_t ctorState)
	{
		/*
	 	* Call generateMode() to bitpack the current state of the Index:
	 	* 8 bits for m_prime + 4 bits for m_ctorState + 20 bits as a counter
	 	*/
		int32_t thisCounter = counter.fetch_add(1);
                return   ((prime & MASK_PRIME) << SHIFT_PRIME)
                       | ((ctorState & MASK_STATE) << SHIFT_STATE)
                       | ((thisCounter & MASK_COUNTER) << SHIFT_COUNTER);
	}
}
