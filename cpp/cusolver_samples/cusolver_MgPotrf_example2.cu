/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverMg.h>

#include "cusolverMg_utils.h"
#include "cusolver_utils.h"

/* compute |x|_inf */
template <typename T> static T vec_nrm_inf(int n, const T *x) {
    T max_nrm = 0.0;
    for (int row = 1; row <= n; row++) {
        T xi = x[IDX1F(row)];
        max_nrm = (max_nrm > fabs(xi)) ? max_nrm : fabs(xi);
    }
    return max_nrm;
}

/* A is 1D laplacian, return A(N:-1:1, :) */
template <typename T> static void gen_1d_laplacian(int N, T *A, int lda) {
    for (int J = 1; J <= N; J++) {
        A[IDX2F(J, J, lda)] = 2.0;
        if ((J - 1) >= 1) {
            A[IDX2F(J, J - 1, lda)] = -1.0;
        }
        if ((J + 1) <= N) {
            A[IDX2F(J, J + 1, lda)] = -1.0;
        }
    }
}

/* Generate matrix B := A * X */
template <typename T>
static void gen_ref_B(int N, int NRHS, double *A, int lda, double *X, int ldx, double *B, int ldb) {
    for (int J = 1; J <= NRHS; J++) {
        for (int I = 1; I <= N; I++) {
            for (int K = 1; K <= N; K++) {
                T Aik = A[IDX2F(I, K, lda)];
                T Xk = X[IDX2F(K, J, ldx)];
                B[IDX2F(I, J, ldb)] += (Aik * Xk);
            }
        }
    }
}

/* Apply inverse to RHS matrix */
template <typename T>
static void solve_system_with_invA(int N, int NRHS, T *A, int lda, T *B, int ldb, T *X, int ldx) {
    /* Extend lower triangular A to full matrix */
    for (int I = 1; I <= N; I++) {
        for (int J = (I + 1); J <= N; J++) {
            A[IDX2F(I, J, lda)] = A[IDX2F(J, I, lda)];
        }
    }

#ifdef SHOW_FORMAT
    std::printf("Full INV (A) = matlab base-1\n");
    print_matrix(N, N, A, lda);
#endif

    /* Reset input matrix to 0 */
    memset(X, 0, sizeof(T) * lda * NRHS);

    /* Apply full inv(A) by matrix-matrix multiplication */
    for (int J = 1; J <= NRHS; J++) {
        for (int I = 1; I <= N; I++) {
            for (int K = 1; K <= N; K++) {
                T Aik = A[IDX2F(I, K, lda)];
                T Bk = B[IDX2F(K, J, ldx)];
                X[IDX2F(I, J, ldx)] += (Aik * Bk);
            }
        }
    }
};

int main(int argc, char *argv[]) {
    cusolverMgHandle_t cusolverH = NULL;

    using data_type = double;

    /* maximum number of GPUs */
    const int MAX_NUM_DEVICES = 16;

    int nbGpus = 0;
    std::vector<int> deviceList(MAX_NUM_DEVICES);

    const int NRHS = 2;
    const int N = 8;

    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; /* tile size of A */
    const int lda = N;
    const int ldb = N;

    int info = 0;

    cudaLibMgMatrixDesc_t descrA;
    cudaLibMgGrid_t gridA;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    int64_t lwork_potrf = 0;
    int64_t lwork_potri = 0;
    int64_t lwork = 0; /* workspace: number of elements per device */

    std::printf("Test 1D Laplacian of order %d\n", N);

    std::printf("Step 1: Create Mg handle and select devices \n");
    CUSOLVER_CHECK(cusolverMgCreate(&cusolverH));

    CUDA_CHECK(cudaGetDeviceCount(&nbGpus));

    nbGpus = (nbGpus < MAX_NUM_DEVICES) ? nbGpus : MAX_NUM_DEVICES;
    std::printf("\tThere are %d GPUs \n", nbGpus);
    for (int j = 0; j < nbGpus; j++) {
        deviceList[j] = j;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, j));
        std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
    }

    CUSOLVER_CHECK(cusolverMgDeviceSelect(cusolverH, nbGpus, deviceList.data()));

    std::printf("Step 2: Enable peer access \n");
    enablePeerAccess(nbGpus, deviceList.data());

    std::printf("Step 3: Allocate host memory A \n");
    std::vector<data_type> A(lda * N, 0);
    std::vector<data_type> B(ldb * NRHS, 0);
    std::vector<data_type> Xref(ldb * NRHS, 0);
    std::vector<data_type> Xans(ldb * NRHS, 0);

    std::printf("Step 4: Prepare 1D Laplacian for A and Xref = ones(N,NRHS) \n");
    gen_1d_laplacian<data_type>(N, &A[IDX2F(IA, JA, lda)], lda);

#ifdef SHOW_FORMAT
    std::printf("A = matlab base-1\n");
    print_matrix(N, N, A.data(), lda);
#endif

    /* X = ones(N,1) */
    for (int row = 1; row <= N; row++) {
        for (int col = 1; col <= NRHS; col++) {
            Xref[IDX2F(row, col, ldb)] = 1.0;
        }
    }

#ifdef SHOW_FORMAT
    std::printf("X = matlab base-1\n");
    print_matrix(N, NRHS, Xref.data(), lda, CUBLAS_OP_T);
#endif

    /* Set B := A * X */
    printf("Step 5: Create RHS for reference solution on host B = A*X \n");
    gen_ref_B<data_type>(N, NRHS, A.data(), /* input */
                         lda, Xref.data(),  /* input */
                         ldb,               /* same leading dimension as B */
                         B.data(),          /* output */
                         ldb);

#ifdef SHOW_FORMAT
    std::printf("B = matlab base-1\n");
    print_matrix(N, NRHS, B.data(), ldb, CUBLAS_OP_T);
#endif

    std::printf("Step 6: Create matrix descriptors for A and D \n");

    CUSOLVER_CHECK(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));

    /* (global) A is N-by-N */
    CUSOLVER_CHECK(cusolverMgCreateMatrixDesc(&descrA, N, /* number of rows of (global) A */
                                              N,          /* number of columns of (global) A */
                                              N,          /* number or rows in a tile */
                                              T_A,        /* number of columns in a tile */
                                              traits<data_type>::cuda_data_type, gridA));

    std::printf("Step 7: Allocate distributed matrices A and B \n");

    std::vector<data_type *> array_d_A(nbGpus, nullptr);
    std::vector<data_type *> array_d_B(nbGpus, nullptr);

    /* A := 0 */
    createMat<data_type>(nbGpus, deviceList.data(), N, /* number of columns of global A */
                         T_A,                          /* number of columns per column tile */
                         lda,                          /* leading dimension of local A */
                         array_d_A.data());

    std::printf("Step 8: Prepare data on devices \n");
    memcpyH2D<data_type>(nbGpus, deviceList.data(), N, N,
                         /* input */
                         A.data(), lda,
                         /* output */
                         N,                /* number of columns of global A */
                         T_A,              /* number of columns per column tile */
                         lda,              /* leading dimension of local A */
                         array_d_A.data(), /* host pointer array of dimension nbGpus */
                         IA, JA);

    std::printf("Step 9: Allocate workspace space \n");
    CUSOLVER_CHECK(
        cusolverMgPotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N,
                                   reinterpret_cast<void **>(array_d_A.data()), IA, /* base-1 */
                                   JA,                                              /* base-1 */
                                   descrA, traits<data_type>::cuda_data_type, &lwork_potrf));

    CUSOLVER_CHECK(cusolverMgPotri_bufferSize(
        cusolverH, CUBLAS_FILL_MODE_LOWER, N, reinterpret_cast<void **>(array_d_A.data()), IA, JA,
        descrA, traits<data_type>::cuda_data_type, &lwork_potri));

    lwork = std::max(lwork_potrf, lwork_potri);
    std::printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));

    std::vector<data_type *> array_d_work(nbGpus, nullptr);

    /* array_d_work[j] points to device workspace of device j */
    workspaceAlloc(nbGpus, deviceList.data(),
                   sizeof(data_type) * lwork, /* number of bytes per device */
                   reinterpret_cast<void **>(array_d_work.data()));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Step 10: Solve A*X = B by POTRF and POTRI \n");
    CUSOLVER_CHECK(cusolverMgPotrf(
        cusolverH, CUBLAS_FILL_MODE_LOWER, N, reinterpret_cast<void **>(array_d_A.data()), IA, JA,
        descrA, traits<data_type>::cuda_data_type, reinterpret_cast<void **>(array_d_work.data()),
        lwork, &info /* host */
        ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if A is singular */
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUSOLVER_CHECK(cusolverMgPotri(
        cusolverH, CUBLAS_FILL_MODE_LOWER, N, reinterpret_cast<void **>(array_d_A.data()), IA, JA,
        descrA, traits<data_type>::cuda_data_type, reinterpret_cast<void **>(array_d_work.data()),
        lwork, &info /* host */
        ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if parameters are valid */
    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("Step 11: Gather INV(A) from devices to host \n");
    memcpyD2H<data_type>(nbGpus, deviceList.data(), N, N,
                         /* input */
                         N,   /* number of columns of global A */
                         T_A, /* number of columns per column tile */
                         ldb, /* leading dimension of local A */
                         array_d_A.data(), IA, JA,
                         /* output */
                         A.data(), /* N-by-N */
                         ldb);

#ifdef SHOW_FORMAT
    /* A is N-by-N */
    std::printf("Computed solution INV(A)\n");
    print_matrix(N, N, A.data(), lda);
#endif

    printf("Step 12: Solve linear system B := inv(A) * B \n");
    solve_system_with_invA(N, NRHS, A.data(), lda, B.data(), ldb, Xans.data(), ldb);

#ifdef SHOW_FORMAT
    /* X is N-by-1*/
    std::printf("Computed solution Xans\n");
    print_matrix(N, NRHS, Xans.data(), ldb);
#endif

    std::printf("Step 13: Measure residual error |Xref - Xans| \n");
    data_type max_err = 0.0;
    for (int col = 1; col <= NRHS; col++) {
        std::printf("errors for X[:,%d] \n", col);
        for (int row = 1; row <= N; row++) {
            data_type Xref_ij = Xref[IDX2F(row, col, ldb)];
            data_type Xans_ij = Xans[IDX2F(row, col, ldb)];
            data_type err = fabs(Xref_ij - Xans_ij);
            max_err = (err > max_err) ? err : max_err;
        }
        data_type Xref_nrm_inf = vec_nrm_inf(N, &Xref[IDX2F(1, col, ldb)]);
        data_type Xans_nrm_inf = vec_nrm_inf(N, &Xans[IDX2F(1, col, ldb)]);
        data_type A_nrm_inf = 4.0;
        data_type rel_err = max_err / (A_nrm_inf * Xans_nrm_inf + Xref_nrm_inf);
        std::printf("\t|b - A*x|_inf = %E\n", max_err);
        std::printf("\t|Xref|_inf = %E\n", Xref_nrm_inf);
        std::printf("\t|Xans|_inf = %E\n", Xans_nrm_inf);
        std::printf("\t|A|_inf = %E\n", A_nrm_inf);
        /* relative error is around machine zero  */
        /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
        std::printf("\t|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);
    }

    std::printf("Step 14: Free resources \n");
    workspaceFree(nbGpus, deviceList.data(), reinterpret_cast<void **>(array_d_work.data()));

    destroyMat(nbGpus, deviceList.data(), N, /* number of columns of global A */
               T_A,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_A.data()));

    CUSOLVER_CHECK(cusolverMgDestroyMatrixDesc(descrA));
    CUSOLVER_CHECK(cusolverMgDestroyGrid(gridA));
    CUSOLVER_CHECK(cusolverMgDestroy(cusolverH));

    return EXIT_SUCCESS;
}
