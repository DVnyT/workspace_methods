#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace ContractID
{
	/*
	 * Call the ContractID::generateID(int cloneNumber, int dim, int prime, bool isCopy)
	 * to generate the m_contractID for an Index with 
	 *
	 *
	 *
	 *
	 *
	 *
	 *
	 */
        static constexpr int BITS_COPY  = 1;
        static constexpr int BITS_PRIME = 8;
        static constexpr int BITS_DIM   = 32;
        static constexpr int BITS_CLONE = 64 - (BITS_COPY + BITS_PRIME + BITS_DIM);
        static_assert(BITS_CLONE > 0, "Not enough bits left!");

        static constexpr int SHIFT_COPY  = 0;
        static constexpr int SHIFT_PRIME = SHIFT_COPY  + BITS_COPY;
        static constexpr int SHIFT_DIM   = SHIFT_PRIME + BITS_PRIME;
        static constexpr int SHIFT_CLONE = SHIFT_DIM   + BITS_DIM;

        static constexpr uint64_t MASK_COPY  = (1ULL << BITS_COPY)  - 1;
        static constexpr uint64_t MASK_PRIME = (1ULL << BITS_PRIME) - 1;
        static constexpr uint64_t MASK_DIM   = (1ULL << BITS_DIM)   - 1;
        static constexpr uint64_t MASK_CLONE = (1ULL << BITS_CLONE) - 1;

        static constexpr uint32_t MAX_PRIME = (1U << BITS_PRIME) - 1;
        static constexpr uint32_t MAX_DIM   = (1ULL << BITS_DIM) - 1;
	
	struct ID {uint64_t raw}; 

        inline ID generateID(uint64_t cloneNumber, uint32_t dim, uint32_t prime, bool isCopy)
        {
                if (cloneNumber > MASK_CLONE)
		{
                        throw std::out_of_range("cloneNumber out of range");
		}

                if (dim > MASK_DIM)
		{
                        throw std::out_of_range("dim out of range");	
		}

                if (prime > MASK_PRIME)
		{
                        throw std::out_of_range("prime out of range");	
		}
                ID id;
                id.raw = ((cloneNumber & MASK_CLONE) << SHIFT_CLONE)
                       | ((uint64_t(dim)   & MASK_DIM)   << SHIFT_DIM)
                       | ((uint64_t(prime) & MASK_PRIME) << SHIFT_PRIME)
                       | ((uint64_t(isCopy) & MASK_COPY) << SHIFT_COPY);
                return id;       
        }

        inline std::ostream& operator<<(std::ostream& os, ID contractID)
        {
                os << "ContractID {Clone = " 
		<< static_cast<unsigned long long> ((contractID.raw >> SHIFT_CLONE) & MASK_CLONE)
                << ", Dim = "   
		<< static_cast<unsigned long long> (uint32_t((contractID.raw >> SHIFT_DIM) & MASK_DIM))
                << ", Prime = " 
		<< static_cast<unsigned long long> (uint32_t((contractID.raw >> SHIFT_PRIME) & MASK_PRIME))
                << ", isCopy = " << (bool((contractID.raw >> SHIFT_COPY) & MASK_COPY) ? "true" : "false")
                << " }";
                return os;
        }
}

