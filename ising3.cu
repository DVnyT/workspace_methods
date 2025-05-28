#include <chrono>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x) {                                                                
  const auto err = x;                                                                    
  if( err != CUTENSOR_STATUS_SUCCESS )                                                   
  { printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); exit(-1); } 
};

// Handle CUDA errors
#define HANDLE_CUDA_ERROR(x) {                                                       
  const auto err = x;                                                                
  if( err != cudaSuccess )                                                           
  { printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__); exit(-1); } 
};

class CPUTimer
{
public:
    void start()
    {
        start_ = std::chrono::steady_clock::now();
    }

    double seconds()
    {
        end_ = std::chrono::steady_clock::now();
        elapsed_ = end_ - start_;
        //return in ms
        return elapsed_.count() * 1000;
    }

private:
    typedef std::chrono::steady_clock::time_point tp;
    tp start_;
    tp end_;
    std::chrono::duration<double> elapsed_;
};

struct GPUTimer
{
  GPUTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
  }

  ~GPUTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start()
  {
    cudaEventRecord(start_, 0);
  }

  float seconds()
  {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
  }
  private:
  cudaEvent_t start_, stop_;
};

int main(){
// 1.a) Like in ITensors, define some indices, say 'a', 'b', 'c' with some dimensions. 
//	We pipe this data into a map,
	std::unordered_map<int, int64_t> extent;
	extent['a'] = 8; // dimension of the index 'a'
	extent['b'] = 8;
	extent['c'] = 16;

// 1.b) Define the 'modes' (indices) of a particular tensor instance A
	std::vector<int> modeA = {'a','b'};

// 1.c) Define a vector that actually holds the dimensions of the indices of A, mapping values from extent!
	std::vector<int64_t> extentA;
	for (auto mode: modeA)
	  extentA.push_back(extent[mode]);

// 1.d) Find the total number of coefficients that go into a tensor A
	size_t elementsA = 1;
	for (auto mode: modeA)
	  elementsA *= extent[mode];
	size_t sizeA = sizeof(floatTypeA) * elementsA; // Think of storing the tensor as a very long array

// 2.a) Allocate memory for a pointer A_d on the device
	void* A_d;
	cudaMalloc((void**)&A_d, sizeA);

// 2.b) Allocate memory for a pointer A on host and fill up the values of the tensor
	floatTypeA* A = (floatType*) malloc(sizeA);

	for (int64_t i = 0; i < elementsA; i++){
		A[i] = ((float) rand())/RAND_MAX + 0.5;
	}

// 2.c) Copy values of A to A_d
	HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));


	const uint32_t kAlignment = 128;
	assert(uintptr_t(A_d) % kAlignment == 0);

// 3) Initializing the cudatensor handles (Recall windowHandler?)
	cutensorHandle_t handle;
	HANDLE_ERROR(cutensorCreate(&handle);

	cutensorTensorDescriptor_t descA;
	HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
					     &descA,
					     nmodeA,
					     extentA.data(),
	        			     NULL,
					     typeA,
					     kAlignment));
// 4) Create Contraction Descriptors
	cutensorOperationDescriptor_t desc;
	HANDLE_ERROR(cutensorCreateContraction(handle,
					&desc,
					descA, modeA.data(), CUTENSOR_OP_IDENTITY,
					descB, modeB.data(), CUTENSOR_OP_IDENTITY,
					descCompute));

// 5) Determine the Algorithm to contract the tensor
	const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
	cutensorPlanPreference_t planpref;
	HANDLE_ERROR(cutensorCreatePlanPreference(handle,
					   &planpref,
					   algo,
					   CUTENSOR_JIT_MODE_DEFAULT));

// 6) Query workspace estimate
	uint64_t workspaceSizeEstimate = 0;
	const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
	HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
					    desc,
					    planPref,
					    workspacePref,
					    &workspaceSizeEstimate));
	cutensorPlan_t plan;
	HANDLE_ERROR(cutensorCreatePlan(handle,
				 &plan,
				 desc,
				 planpref,
				 workspaceSizeEstimate));
	uint64_t actualWorkspaceSize = 0;
	HANDLE_ERROR(cutensorPlanGetAttribute(handle,
    					plan,
    					CUTENSOR_PLAN_REQUIRED_WORKSPACE,
    					&actualWorkspaceSize,
    					sizeof(actualWorkspaceSize)));

	assert(actualWorkspaceSize <= workspaceSizeEstimate);
	void* work = nullptr;
	if(actualWorkspaceSize > 0){
		HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
		assert(uintptr_t(work) % 128 == 0);
	}

// 7) EXECUTE!
	cudaStream_t stream;
	HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
	HANDLE_ERROR(cutensorContract(handle,
				plan,
				(void*) &alpha, A_d, B_d,
				(void*) &beta,  C_d, C_d, 
                		work, actualWorkspaceSize, stream));
	return 0;	

}
