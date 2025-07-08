#include "../include/CudaUtils.hpp"
#include "../include/DevicePools.hpp"
#include "../include/Index.hpp"
#include "../include/Tensor.hpp"
#include <array>
#include <chrono>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

int main()
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cutensorHandle_t handle;
	cutensorCreate(&handle);
        Index a(3), b(2), c(2);
        Tensor A{std::vector<Index>{a, b}};
        auto ptr = A.getHostPtr();
        for (int i = 0; i < A.getElements(); i++)
        {
                ptr[i] = i;
        }
	
        Tensor B{std::vector<Index>{b, c}};
	auto ptr2 = B.getHostPtr();
        for (int i = 0; i < B.getElements(); i++)
        {
                ptr2[i] = i + 5;
        }
        
	Tensor C = contractAB(A, B, handle, stream);
        for (int i = 0; i < C.getElements(); i++)
        {
                std::cout << C.getHostPtr()[i] << " ";
        }

        return 0;
}
