#include "catch_amalgamated.hpp"

#include "../include/CudaUtils.hpp"
#include "../include/DevicePools.hpp"
#include "../include/Index.hpp"
#include "../include/Tensor.hpp"
#include <vector>

// HELPERS =>
// 1. Helper function to create test indices using your Index constructors
std::vector<Index> createTestIndices(const std::vector<int64_t>& extents)
{
        std::vector<Index> indices;
        for (const auto& extent : extents)
        {
                indices.emplace_back(extent);
        }
        return indices;
}

// 2. Helper function to create test indices with specific primes
std::vector<Index> createTestIndicesWithPrimes(const std::vector<std::pair<int64_t, int>>& extent_prime_pairs)
{
        std::vector<Index> indices;
        for (const auto& [extent, prime] : extent_prime_pairs)
        {
                indices.emplace_back(extent, prime);
        }
        return indices;
}

// 3. Helper function to fill a host tensor with 1,2,3,... then copy to device
void fillSequence(Tensor& T, cudaStream_t s)
{
        auto ptr = T.getHostPtr();
        for (size_t i = 0; i < T.getElements(); ++i)
                ptr[i] = static_cast<floatType>(i + 1);
        T.cpyToDevice(s);
}

// TEST_CASES =>

TEST_CASE("Scaling Kernels")
{
        CudaStream stream;

        // make a 3x2 matrix A in tensor form: order=2, extents={3, 2}
        Tensor U({Index(3), Index(2)});
        auto tmp = U.getHostPtr();
        for (int i = 0; i < U.getElements(); i++)
        {
                tmp[i] = i + 1;
        }
        U.cpyToDevice(stream); // A = [1 2; 3 4; 5 6] in column-major

        // singular values vector s of length min(2,3)=2
        Tensor S({Index(2)});

        // fill with [10, 100]
        tmp = S.getHostPtr();
        tmp[0] = 10.f;
        tmp[1] = 100.f;
        S.cpyToDevice(stream);

        // scale rows of A by S => row 0 *=10, row1 *=100
        scaleUOnDevice(reinterpret_cast<floatType*>(U.getDevicePtr()),
                       reinterpret_cast<const floatType*>(S.getDevicePtr()), 3, 2, stream);
        U.cpyToHost(stream);
        cudaStreamSynchronize(stream);

        // After scaling: row0 of each column (i=0) multiplied by 10, row1 by 100
        // original column-major data: [1; 3; 5], [2; 4; 6]
        // new: [1; 3; 5] => [10; 30; 50], [2; 4; 6] => [200, 400, 600] 
        tmp = U.getHostPtr();

        for (size_t i = 0; i < 6; ++i)
        {
        	std::cout << tmp[i] << " ";
	}
	std::cout << std::endl;
        std::vector<floatType> expect = {10, 30, 50, 200, 400, 600};
        for (size_t i = 0; i < 6; ++i)
        {
                REQUIRE(tmp[i] == Catch::Approx(expect[i]));
        }
}

TEST_CASE("Tensor Construction")
{
        SECTION("Construction with Indices")
        {
                std::vector<Index> indices = createTestIndices({2, 3, 4});

                Tensor A(indices);

                REQUIRE(A.getOrder() == 3);
                REQUIRE(A.getElements() == 2 * 3 * 4);
                REQUIRE(A.getExtents().size() == 3);
                REQUIRE(A.getExtents()[0] == 2);
                REQUIRE(A.getExtents()[1] == 3);
                REQUIRE(A.getExtents()[2] == 4);
                REQUIRE(A.getByteSize() == 24 * sizeof(floatType));
                REQUIRE(A.getHostPtr() != nullptr);
                REQUIRE(A.getDevicePtr() != nullptr);
        }

        SECTION("Construction with Primed Indices")
        {
                std::vector<Index> indices = createTestIndicesWithPrimes({{2, 0}, {3, 1}, {4, 2}});
                Tensor B(indices);

                REQUIRE(B.getOrder() == 3);
                REQUIRE(B.getElements() == 24);

                // Check that the indices were properly set
                const auto& inds = B.getInds();
                REQUIRE(inds.size() == 3);
                REQUIRE(inds[0].getPrime() == 0);
                REQUIRE(inds[1].getPrime() == 1);
                REQUIRE(inds[2].getPrime() == 2);
        }
}

TEST_CASE("Tensor Memory Management")
{
        CudaStream stream;
        SECTION("Memory initialization")
        {
                std::vector<Index> inds = createTestIndices({2, 3});
                Tensor A(inds);
                REQUIRE(A.getHostPtr() != nullptr);
                REQUIRE(A.getDevicePtr() != nullptr);
        }

        SECTION("Host-Device memory transfer")
        {
                std::vector<Index> inds = createTestIndices({40, 200});
                Tensor tensor(inds);

                // Set some values on host (column-major ordering)
                auto hostPtr = tensor.getHostPtr();
                for (size_t i = 0; i < tensor.getElements(); ++i)
                {
                        hostPtr[i] = static_cast<floatType>(i + 1);
                }

                // Copy to Device from Host
                tensor.cpyToDevice(stream);

                cudaStreamSynchronize(stream);
                // Zero out host memory
                for (size_t i = 0; i < tensor.getElements(); ++i)
                {
                        hostPtr[i] = 0.0;
                }

                // Copy to Host from Device
                tensor.cpyToHost(stream);
                cudaStreamSynchronize(stream);
                // Check values
                for (size_t i = 0; i < tensor.getElements(); ++i)
                {
                        REQUIRE(hostPtr[i] == (static_cast<floatType>(i + 1)));
                }
        }
        // Check that we can't accidentally hold a dangling pointer to memory!
}

TEST_CASE("Tensor Modifiers")
{
        SECTION("setInds")
        {
                std::vector<Index> original_indices = createTestIndices({2, 3});
                std::vector<Index> new_indices = createTestIndices({4, 5});

                Tensor A(original_indices);

                REQUIRE(A.getOrder() == 2);
                REQUIRE(A.getElements() == 6);

                A.setInds(new_indices);

                REQUIRE(A.getOrder() == 2);
                REQUIRE(A.getElements() == 20);
                REQUIRE(A.getExtents()[0] == 4);
                REQUIRE(A.getExtents()[1] == 5);
        }

        SECTION("primeAll")
        {
                std::vector<Index> inds = createTestIndices({2, 3});
                Tensor B(inds);

                // Get original primes
                const auto& originalInds = B.getInds();
                std::vector<int> originalPrimes;
                for (const auto& idx : originalInds)
                {
                        originalPrimes.push_back(idx.getPrime());
                }

                B.primeAll();

                // Check that all indices have been primed
                const auto& primedInds = B.getInds();
                for (size_t i = 0; i < primedInds.size(); ++i)
                {
                        REQUIRE(primedInds[i].getPrime() == originalPrimes[i] + 1);
                }
        }
}

TEST_CASE("Tensor Contractions")
{
        CudaStream stream;
        CutensorHandle handle;
        SECTION("Basic Contract Logic")
        {
                Index a(3), b(2), c(2);

                Tensor A{std::vector<Index>{a, b}};
                for (int i = 0; i < A.getElements(); i++)
                {
                        A.getHostPtr()[i] = i;
                }
                Tensor B{std::vector<Index>{b, c}};
                for (int i = 0; i < B.getElements(); i++)
                {
                        B.getHostPtr()[i] = i + 5;
                }
                assert(A.getDevicePtr() && B.getDevicePtr());
                Tensor C = contractAB(A, B, handle, stream);

                REQUIRE(C.getExtents()[0] == 3);
                REQUIRE(C.getExtents()[1] == 2);
                REQUIRE(C.getOrder() == 2);
                REQUIRE(C.getElements() == 6);
                REQUIRE(C.getByteSize() == sizeof(floatType) * 6);
                for (int i = 0; i < C.getElements(); i++)
                {
                        std::cout << C.getHostPtr()[i] << " ";
                }
                std::cout << std::endl;
                std::cout << A << '\n' << B << '\n' << C << std::endl;
        }

        SECTION("Get unique indices")
        {
                Index a(3), b(4), c(5, 1), d(3);
                Index e(a);
                Index f = b;
                Index g(c);
                g.setExtent(8);
                Index h(d);
                h.prime();

                Tensor A({a, b, c, d});
                Tensor B({e, f, g, h});

                // We expect,
                // a and e contract
                // b and f contract
                // c(5) and g(8) are unique
                // d(3) and h(3) are unique

                auto [uniqueModesA, uniqueExtentsA] = getUniqueIndsAB(A, B);
                // Though we won't know the modes (large 32 bit number), the extents should come out to be,
                // 5, 8, 3, 3
        }
}

TEST_CASE("Tensor SVD Operations")
{
        CudaStream stream;
        CusolverHandle handle;
        SECTION("Left SVD")
        {
                std::vector<Index> indices = createTestIndices({2, 2});
                Tensor A(indices);

                A.initOnHost();
                A.initOnDevice();

                // Initialize with some values
                floatType* ptr = A.getHostPtr();
                ptr[0] = 0;
		ptr[1] = 2;
		ptr[2] = 2;
		ptr[3] = 1;
		
		auto [X, Y, Z] = lSVD(A, 1, handle, stream);
		std::cout << X << '\n' << Y << '\n' << Z << std::endl;
        }
}
