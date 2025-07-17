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
        for (int i = 0; i < T.getElements(); ++i)
                ptr[i] = static_cast<floatType>(i + 1);
        T.cpyToDevice(s);
}

// TEST_CASES =>

TEST_CASE("Scaling Kernels")
{
        CudaStream stream;

        SECTION("Scale U by S: Postmultiply U")
        {
                // Make a 3 x 2 matrix U in tensor form: order = 2, extents = {3, 2}
                Tensor U({Index(3), Index(2)});
                auto tmp1 = U.getHostPtr();
                for (int i = 0; i < U.getElements(); i++)
                {
                        tmp1[i] = i + 1;
                }

                U.cpyToDevice(stream); // A = [1 4] in column-major
                                       //     [2 5]
                                       //     [3 6]

                // Singular values vector s of length min(3, 2) = 2
                Tensor S({Index(2)});

                // Say the singular values are [10, 100]
                auto tmp2 = S.getHostPtr();
                tmp2[0] = 10.f;
                tmp2[1] = 100.f;
                S.cpyToDevice(stream);

                // Scale columns of U by S => col0 *= 10, col1 *= 100
                scaleUOnDevice(reinterpret_cast<floatType*>(U.getDevicePtr()),
                               reinterpret_cast<floatType*>(S.getDevicePtr()), 3, 2, stream);
                U.cpyToHost(stream);
                cudaStreamSynchronize(stream);

                // [1; 2; 3] => [10; 20; 30], [4; 5; 6] => [400, 500, 600]
                std::vector<floatType> expect = {10, 20, 30, 400, 500, 600};
                for (int i = 0; i < U.getElements(); ++i)
                {
                        REQUIRE(tmp1[i] == Catch::Approx(expect[i]));
                }
        }

        SECTION("Scale Vd by S: Premultiply Vd")
        {
                // Make a 2 x 3 matrix Vd in tensor form: order = 2, extents = {2, 3}
                Tensor Vd({Index(2), Index(3)});
                auto tmp1 = Vd.getHostPtr();
                for (int i = 0; i < Vd.getElements(); i++)
                {
                        tmp1[i] = i + 1;
                }

                Vd.cpyToDevice(stream); // Vd = [1 3 5] in column-major
                                        //      [2 4 6]

                // Singular values vector s of length min(2, 3) = 2
                Tensor S({Index(2)});

                // Say the singular values are [10, 100]
                auto tmp2 = S.getHostPtr();
                tmp2[0] = 10.f;
                tmp2[1] = 100.f;
                S.cpyToDevice(stream);

                // Scale rows of Vd by S => row0 *= 10, row1 *= 100
                scaleVdOnDevice(reinterpret_cast<floatType*>(Vd.getDevicePtr()),
                               reinterpret_cast<floatType*>(S.getDevicePtr()), 2, 3, stream);
                Vd.cpyToHost(stream);
                cudaStreamSynchronize(stream);

                // [1 3 5] => [10 30 50]; [2 4 6] => [200 400 600]
                std::vector<floatType> expect = {10, 200, 30, 400, 50, 600};
                for (int i = 0; i < Vd.getElements(); ++i)
                {
                        REQUIRE(tmp1[i] == Catch::Approx(expect[i]));
                }
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
                
		REQUIRE(B.getOrder() == 3);
                REQUIRE(B.getElements() == 2 * 3 * 4);
                REQUIRE(B.getExtents().size() == 3);
                REQUIRE(B.getExtents()[0] == 2);
                REQUIRE(B.getExtents()[1] == 3);
                REQUIRE(B.getExtents()[2] == 4);
                REQUIRE(B.getByteSize() == 24 * sizeof(floatType));
                REQUIRE(B.getHostPtr() != nullptr);
                REQUIRE(B.getDevicePtr() != nullptr);
        }
}

TEST_CASE("Tensor Memory Management")
{
        CudaStream stream;

        SECTION("Host-Device memory transfer")
        {
                std::vector<Index> inds = createTestIndices({4, 20});
                Tensor tensor(inds);

                // Set some values on host (column-major ordering)
                auto hostPtr = tensor.getHostPtr();
                for (int i = 0; i < tensor.getElements(); ++i)
                {
                        hostPtr[i] = static_cast<floatType>(i + 1);
                }

                // Copy to Device from Host
                tensor.cpyToDevice(stream);
		cudaStreamSynchronize(stream);
                
		// Zero out host memory
                for (int i = 0; i < tensor.getElements(); ++i)
                {
                        hostPtr[i] = 0.0;
                }

                // Copy to Host from Device
                tensor.cpyToHost(stream);
		cudaStreamSynchronize(stream);
                
		// Check values
                for (int i = 0; i < tensor.getElements(); ++i)
                {
                        REQUIRE(hostPtr[i] == (static_cast<floatType>(i + 1)));
                }
        }
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
                for (int i = 0; i < primedInds.size(); ++i)
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
                std::vector<Index> inds = createTestIndices({2, 2});
                Tensor A(inds);

                A.initOnHost();
                A.initOnDevice();

                // Initialize with some values
                floatType* ptr = A.getHostPtr();
                ptr[0] = 0; // [0, 3]
                ptr[1] = 2; // [2, 1]. Data is written in column-major format.
                ptr[2] = 3;
                ptr[3] = 1;

                auto [U, Vd] = lSVD(A, 1, handle, stream);
                std::cout << U << '\n' << Vd << '\n' << std::endl;
        }
        SECTION("Right SVD")
        {
                std::vector<Index> inds = createTestIndices({2, 2});
                Tensor B(inds);

                B.initOnHost();
                B.initOnDevice();

                // Initialize with some values
                floatType* ptr = B.getHostPtr();
                ptr[0] = 0; // [0, 3]
                ptr[1] = 2; // [2, 1]. Data is written in column-major format.
                ptr[2] = 3;
                ptr[3] = 1;

                auto [U, Vd] = rSVD(B, 1, handle, stream);
                std::cout << U << '\n' << Vd << '\n' << std::endl;
        }
}
