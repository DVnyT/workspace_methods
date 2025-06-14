#include <stdio.h>

#include <unordered_map>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
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

int main()
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeB = CUTENSOR_R_32F;
    cutensorDataType_t typeC = CUTENSOR_R_32F;
    const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /**********************
     * Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v}
     **********************/

    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 96;
    extent['n'] = 96;
    extent['u'] = 96;
    extent['v'] = 64;
    extent['h'] = 64;
    extent['k'] = 64;

    double gflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['k'] * extent['h']) /1e9;

    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    void *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Error: Host allocation of A or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsA; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsB; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int64_t i = 0; i < elementsC; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;

    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Load plan cache
     **********************/

    // holds information about the per-handle plan cache
    const char planCacheFilename[] = "./planCache.bin";
    uint32_t numCachelines = 0;
    cutensorStatus_t status = cutensorHandleReadPlanCacheFromFile(handle,
            planCacheFilename, &numCachelines);
    if (status == CUTENSOR_STATUS_IO_ERROR)
    {
        printf("File (%s) doesn't seem to exist.\n", planCacheFilename);
    }
    else if (status != CUTENSOR_STATUS_SUCCESS)
    {
        printf("cutensorHandleReadPlanCacheFromFile reports error: %s\n", cutensorGetErrorString(status));
    }
    else
    {
        printf("cutensorHandleReadPlanCacheFromFile read %d cachelines from file.\n",
                numCachelines);
    }

    /**********************
     * Optional: Resize the cache in case you expect the default option to be insufficient fore your use case
     **********************/
    uint32_t numEntries = 128;
    HANDLE_ERROR(cutensorHandleResizePlanCache(handle, numEntries));

    /**********************
     * Create Tensor Descriptors
     **********************/
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, kAlignment));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, 
                 &desc,
                 descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                 descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(),
                 descCompute));

    /**************************
     * PlanPreference: Set the algorithm to use and enable incremental autotuning
     ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                               handle,
                               &planPref,
                               algo,
                               CUTENSOR_JIT_MODE_NONE)); // disable just-in-time compilation

    const cutensorCacheMode_t cacheMode = CUTENSOR_CACHE_MODE_PEDANTIC;
    HANDLE_ERROR(cutensorPlanPreferenceSetAttribute(
        handle,
        planPref,
        CUTENSOR_PLAN_PREFERENCE_CACHE_MODE,
        &cacheMode,
        sizeof(cutensorCacheMode_t)));

    const cutensorAutotuneMode_t autotuneMode = CUTENSOR_AUTOTUNE_MODE_INCREMENTAL;
    HANDLE_ERROR(cutensorPlanPreferenceSetAttribute(
        handle,
        planPref,
        CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE,
        &autotuneMode ,
        sizeof(cutensorAutotuneMode_t)));

    const uint32_t incCount = 4;
    HANDLE_ERROR(cutensorPlanPreferenceSetAttribute(
        handle,
        planPref,
        CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT,
        &incCount,
        sizeof(uint32_t)));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                                          desc,
                                          planPref,
                                          workspacePref,
                                          &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                 &plan,
                 desc,
                 planPref,
                 workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
        plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize,
        sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    for (int i=0; i < incCount + 1; ++i) // last iteration will hit the cache
    {
        cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // Set up timing
        GPUTimer timer;
        timer.start();

        // Automatically takes advantage of the incremental-autotuning (and updates the cache inside the context)
        HANDLE_ERROR(cutensorContract(handle,
                                  plan,
                                  (void*) &alpha, A_d, B_d,
                                  (void*) &beta,  C_d, C_d, 
                                  work, actualWorkspaceSize, 0 /* stream */));

        // Synchronize and measure timing
        auto time = timer.seconds();

        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }

    /*************************/

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", gflops / minTimeCUTENSOR, transferedBytes/ minTimeCUTENSOR);

    status = cutensorHandleWritePlanCacheToFile(handle, planCacheFilename);
    if (status == CUTENSOR_STATUS_IO_ERROR)
    {
        printf("File (%s) couldn't be written to.\n", planCacheFilename);
    }
    else if (status != CUTENSOR_STATUS_SUCCESS)
    {
        printf("cutensorHandleWritePlanCacheToFile reports error: %s\n",
                cutensorGetErrorString(status));
    }
    else
    {
        printf("Plan cache successfully stored to %s.\n", planCacheFilename);
    }


    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}
