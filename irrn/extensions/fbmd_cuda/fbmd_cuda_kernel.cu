#include "fbmd_cuda.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusolver_common.h>

/**********************  Batch SymEig **************************************/

// Functions that compute the buffer size 

#define SYEVJ_BUFFER_ARGTYPES(T1, T2)                                        \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,\
    int m, const T1 *A, int lda, const T2 *E, int *lwork,                    \
    syevjInfo_t syevj_params, int batch_size


template <typename T1, typename T2>
cusolverStatus_t _syevjBatched_bufferSize(
    SYEVJ_BUFFER_ARGTYPES(T1, T2));

template <>
cusolverStatus_t _syevjBatched_bufferSize<float, float>(
    SYEVJ_BUFFER_ARGTYPES(float, float)){
    return cusolverDnSsyevjBatched_bufferSize(
            handle, jobz, uplo, m, A, lda, E, lwork, syevj_params, batch_size);
}

template <>
cusolverStatus_t _syevjBatched_bufferSize<double, double>(
    SYEVJ_BUFFER_ARGTYPES(double, double)){
    return cusolverDnDsyevjBatched_bufferSize(
            handle, jobz, uplo, m, A, lda, E, lwork, syevj_params, batch_size);
    }

template <>
cusolverStatus_t _syevjBatched_bufferSize<c10::complex<float>, float>(
    SYEVJ_BUFFER_ARGTYPES(c10::complex<float>, float)){
    return cusolverDnCheevjBatched_bufferSize(
            handle, jobz, uplo, m, reinterpret_cast<const cuComplex*>(A),
            lda, E, lwork, syevj_params, batch_size);
    }
    
template <>
cusolverStatus_t _syevjBatched_bufferSize<c10::complex<double>, double>(
    SYEVJ_BUFFER_ARGTYPES(c10::complex<double>, double)){
    return cusolverDnZheevjBatched_bufferSize(
            handle, jobz, uplo, m, reinterpret_cast<const cuDoubleComplex*>(A),
            lda, E, lwork, syevj_params, batch_size);
    }

// Functions that compute the eigenvectors and eigenvalues

#define SYEVJ_ARGTYPES(T1, T2)                                               \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,\
    int m, T1 *A, int lda, T2 *E, T1 *work, int lwork, int *info,            \
    syevjInfo_t syevj_params, int batch_size

template <typename T1, typename T2>
cusolverStatus_t _syevj_batched_solver(SYEVJ_ARGTYPES(T1, T2));

template <>
cusolverStatus_t _syevj_batched_solver<float, float>(
    SYEVJ_ARGTYPES(float, float)) {
    return cusolverDnSsyevjBatched(
            handle, jobz, uplo, m, A, lda, E, work, lwork, info, 
            syevj_params, batch_size);
    }

template <>
cusolverStatus_t _syevj_batched_solver<double, double>(
    SYEVJ_ARGTYPES(double, double)) {
    return cusolverDnDsyevjBatched(
            handle, jobz, uplo, m, A, lda, E, work, lwork, info, 
            syevj_params, batch_size);
    }

template <>
cusolverStatus_t _syevj_batched_solver<c10::complex<float>, float>(
    SYEVJ_ARGTYPES(c10::complex<float>, float)) {
    return cusolverDnCheevjBatched(
            handle, jobz, uplo, m, reinterpret_cast<cuComplex*>(A), lda, E,
            reinterpret_cast<cuComplex*>(work), lwork, info, syevj_params,
            batch_size);
    }
    
template <>
cusolverStatus_t _syevj_batched_solver<c10::complex<double>, double>(
    SYEVJ_ARGTYPES(c10::complex<double>, double)) {
    return cusolverDnZheevjBatched(
            handle, jobz, uplo, m, reinterpret_cast<cuDoubleComplex*>(A),
            lda, E, reinterpret_cast<cuDoubleComplex*>(work), lwork, info,
            syevj_params, batch_size);
    }

template <typename T1, typename T2>
void _batch_flattened_symeig_cuda_helper(
    T1 *d_A, T2 *d_E, int *info, const int batch_size, const int m,
    const int lda, const int sort, bool eigenvectors, bool upper,
    double tol=1e-7, int max_sweeps=100)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;
    
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    int lwork = 0; // size of workspace
    T1 *d_work = NULL; // device workspace for gesvdjBatched

    // Define whether to compute or not the eigenvectors
    const cusolverEigMode_t jobz = 
        eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    // Define whether to use the upper or lower triangular part of the tensor
    const cublasFillMode_t  uplo = 
        upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    
    // Create cusolver handle, bind a stream  
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // Configuration of syevj   
    status = cusolverDnCreateSyevjInfo(&syevj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);
  
    status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // Enable or disable sorting of the eigenvalues
    status = cusolverDnXsyevjSetSortEig(syevj_params, sort);
    assert(CUSOLVER_STATUS_SUCCESS == status);    
    
    // Query working space of syevjBatched 
    status = _syevjBatched_bufferSize<T1, T2>(
        cusolverH, jobz, uplo, m, d_A, lda, d_E, &lwork, syevj_params,
        batch_size);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // Allocate work array
    cudaStat = cudaMalloc((void**)&d_work, sizeof(T1)*lwork);
    assert(cudaStat == cudaSuccess);

    // Compute eigenvectors and eigenvalues
    status = _syevj_batched_solver<T1, T2>(
        cusolverH, jobz, uplo, m, d_A, lda, d_E, d_work, lwork, info, 
        syevj_params, batch_size);
    cudaStat = cudaDeviceSynchronize();            
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaStat == cudaSuccess);
    
    // free resources
    if (d_work) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream) cudaStreamDestroy(stream);
    if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);    
}

std::tuple<at::Tensor, at::Tensor> _batch_flattened_symeig_cuda(
        at::Tensor &d_A, int sort, bool eigenvectors, bool upper, double tol, 
        int max_sweeps) {
    
    // Transform the tensor to a column-major layout so as to be compatible 
    // with CUBLAS.
    d_A = d_A.transpose(1,2).contiguous().transpose(1,2);
    
    const auto batch_size = d_A.size(0);
    const auto m = d_A.size(1);
    const auto lda = m;

    auto Dtype = (d_A.scalar_type() == torch::ScalarType::Float) ||
                 (d_A.scalar_type() == torch::ScalarType::ComplexFloat)
                 ? at::kFloat : at::kDouble;

    at::Tensor d_E = torch::empty({batch_size, m}, d_A.options().dtype(Dtype));
    at::Tensor info = torch::empty({batch_size}, d_A.options().dtype(at::kInt));

    switch (d_A.scalar_type()) {
        case torch::ScalarType::Double:
            _batch_flattened_symeig_cuda_helper<double>(
                d_A.data_ptr<double>(),
                d_E.data_ptr<double>(),
                info.data_ptr<int>(), batch_size, m, lda, sort, 
                eigenvectors, upper, tol, max_sweeps);
            break;

        case torch::ScalarType::Float:
            _batch_flattened_symeig_cuda_helper<float>(
                d_A.data_ptr<float>(),
                d_E.data_ptr<float>(),
                info.data_ptr<int>(), batch_size, m, lda, sort, 
                eigenvectors, upper, tol, max_sweeps);
            break;
        
        case torch::ScalarType::ComplexDouble:
            _batch_flattened_symeig_cuda_helper<c10::complex<double>, double>(
                d_A.data_ptr<c10::complex<double>>(),
                d_E.data_ptr<double>(),
                info.data_ptr<int>(), batch_size, m, lda, sort, 
                eigenvectors, upper, tol, max_sweeps);
            break;
            
        case torch::ScalarType::ComplexFloat:
            _batch_flattened_symeig_cuda_helper<c10::complex<float>, float>(
                d_A.data_ptr<c10::complex<float>>(),
                d_E.data_ptr<float>(),
                info.data_ptr<int>(), batch_size, m, lda, sort, 
                eigenvectors, upper, tol, max_sweeps);
            break;        

        default:
            AT_ERROR("This function doesn't support types other than "
                     "float, double, complexFloat, and complexDouble.");
    }

    // Check error status
    if (info.ne(0).any().item().toInt()){
        TORCH_WARN("CUSolver (syevj) did not converge");
    }

    return std::make_tuple(d_E, d_A);
}


/**********************  Batch SVD *****************************************/
    
// Functions that compute the buffer size 

#define GESVDJ_BUFFER_ARGTYPES(T1, T2)                                       \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,         \
    const T1 *A, int lda, const T2 *S, const T1 *U, int ldu, const T1 *V,    \
    int ldv, int *lwork, gesvdjInfo_t gesvdj_params, int batch_size          \
    
template <typename T1, typename T2>
cusolverStatus_t _gesvdjBatched_bufferSize(
    GESVDJ_BUFFER_ARGTYPES(T1, T2));

template <>
cusolverStatus_t _gesvdjBatched_bufferSize<float, float>(
    GESVDJ_BUFFER_ARGTYPES(float, float)){
    return cusolverDnSgesvdjBatched_bufferSize(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, 
        gesvdj_params, batch_size);
    }

template <>
cusolverStatus_t _gesvdjBatched_bufferSize<double, double>(
    GESVDJ_BUFFER_ARGTYPES(double, double)){
    return cusolverDnDgesvdjBatched_bufferSize(
        handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, 
        gesvdj_params, batch_size);
    }

template <>
cusolverStatus_t _gesvdjBatched_bufferSize<c10::complex<float>, float>(
    GESVDJ_BUFFER_ARGTYPES(c10::complex<float>, float)){
    return cusolverDnCgesvdjBatched_bufferSize(
        handle, jobz, m, n, reinterpret_cast<const cuComplex*>(A), lda, S,
        reinterpret_cast<const cuComplex*>(U), ldu,
        reinterpret_cast<const cuComplex*>(V), ldv, lwork, gesvdj_params,
        batch_size);
    }

template <>
cusolverStatus_t _gesvdjBatched_bufferSize<c10::complex<double>, double>(
    GESVDJ_BUFFER_ARGTYPES(c10::complex<double>, double)){
    return cusolverDnZgesvdjBatched_bufferSize(
        handle, jobz, m, n, reinterpret_cast<const cuDoubleComplex*>(A), lda,
        S, reinterpret_cast<const cuDoubleComplex*>(U), ldu,
        reinterpret_cast<const cuDoubleComplex*>(V), ldv, lwork, gesvdj_params,
        batch_size);
    }

// Functions for computing the singular vectors and singular values

#define GESVDJ_ARGTYPES(T1, T2)                                              \
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n,         \
    T1 *A, int lda, T2 *S, T1 *U, int ldu, T1 *V, int ldv, T1 *work,         \
    int lwork, int *info, gesvdjInfo_t gesvdj_params, int batch_size         \

template <typename T1, typename T2>
cusolverStatus_t _gesvdj_batched_solver(GESVDJ_ARGTYPES(T1, T2));

template <>
cusolverStatus_t _gesvdj_batched_solver<float, float>(
    GESVDJ_ARGTYPES(float, float)){
    return cusolverDnSgesvdjBatched(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,
            info, gesvdj_params, batch_size);
    }

template <>
cusolverStatus_t _gesvdj_batched_solver<double, double>(
    GESVDJ_ARGTYPES(double, double)){
    return cusolverDnDgesvdjBatched(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,
            info, gesvdj_params, batch_size);
    }
    
template <>
cusolverStatus_t _gesvdj_batched_solver<c10::complex<float>, float>(
    GESVDJ_ARGTYPES(c10::complex<float>, float)){
    return cusolverDnCgesvdjBatched(
            handle, jobz, m, n, reinterpret_cast<cuComplex*>(A), lda, S,
            reinterpret_cast<cuComplex*>(U), ldu,
            reinterpret_cast<cuComplex*>(V), ldv,
            reinterpret_cast<cuComplex*>(work), lwork, info, gesvdj_params,
            batch_size);
    }

template <>
cusolverStatus_t _gesvdj_batched_solver<c10::complex<double>, double>(
    GESVDJ_ARGTYPES(c10::complex<double>, double)){
    return cusolverDnZgesvdjBatched(
            handle, jobz, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, S,
            reinterpret_cast<cuDoubleComplex*>(U), ldu,
            reinterpret_cast<cuDoubleComplex*>(V), ldv,
            reinterpret_cast<cuDoubleComplex*>(work), lwork, info,
            gesvdj_params, batch_size);
    }
    

template <typename T1, typename T2>
void _batch_flattened_svd_cuda_helper(
    T1 *d_A, T2 *d_S, T1 *d_U, T1 *d_V, int *info, const int batch_size,
    const int m, const int n, const int lda, const int ldu, const int ldv,
    const int sort_svd, bool compute_uv, double tol=1e-7, int max_sweeps=100)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;
    
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
    int lwork = 0; // size of workspace
    T1 *d_work = NULL; // device workspace for gesvdjBatched

    // Define whether to compute or not the singular vectors
    const cusolverEigMode_t jobz = 
        compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    // Create cusolver handle, bind a stream  
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // Configuration of gesvdj   
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);
  
    status = cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // Enable or disable sorting of the singular values
    status = cusolverDnXgesvdjSetSortEig(gesvdj_params, sort_svd);
    assert(CUSOLVER_STATUS_SUCCESS == status);    
    
    // Query working space of gesvdjBatched 
    status = _gesvdjBatched_bufferSize<T1, T2>(
            cusolverH, jobz, m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv, 
            &lwork, gesvdj_params, batch_size);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    // Allocate work array
    cudaStat = cudaMalloc((void**)&d_work, sizeof(T1)*lwork);
    assert(cudaStat == cudaSuccess);

    // Compute singular vectors and singular values
    status = _gesvdj_batched_solver<T1, T2>(
            cusolverH, jobz, m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv, 
            d_work, lwork, info, gesvdj_params, batch_size);
    cudaStat = cudaDeviceSynchronize();            
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaStat == cudaSuccess);
    
    // free resources
    if (d_work) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);    
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _batch_flattened_svd_cuda(
        at::Tensor &A, int sort_svd, bool compute_uv, double tol,
        int max_sweeps) {
    
    // Transform tensor to a column-major layout so as to be compatible with CUBLAS.
    A = A.transpose(1,2).contiguous().transpose(1,2);
    
    const auto batch_size = A.size(0);
    const auto m = A.size(1);
    const auto n = A.size(2);
    const auto lda = m;
    const auto min_mn = (m < n)? m : n;
    const auto ldu = m;
    const auto ldv = n;

    auto Dtype = (A.scalar_type() == torch::ScalarType::Float) ||
                 (A.scalar_type() == torch::ScalarType::ComplexFloat)
                 ? at::kFloat : at::kDouble;

    at::Tensor d_U = torch::empty({batch_size, ldu, m}, A.options());
    at::Tensor d_S = torch::empty({batch_size, min_mn}, A.options().dtype(Dtype));
    at::Tensor d_V = torch::empty({batch_size, ldv, n}, A.options());
    at::Tensor info = torch::empty({batch_size}, A.options().dtype(at::kInt));

    switch (A.scalar_type()) {
        case torch::ScalarType::Double:
            _batch_flattened_svd_cuda_helper<double>(
                A.data_ptr<double>(),
                d_S.data_ptr<double>(),
                d_U.data_ptr<double>(),
                d_V.data_ptr<double>(),
                info.data_ptr<int>(), batch_size, m, n, lda, ldu, ldv, 
    		    sort_svd, compute_uv, tol, max_sweeps);
            break;

        case torch::ScalarType::Float:
            _batch_flattened_svd_cuda_helper<float>(
                A.data_ptr<float>(),
                d_S.data_ptr<float>(),
                d_U.data_ptr<float>(),
                d_V.data_ptr<float>(),
                info.data_ptr<int>(), batch_size, m, n, lda, ldu, ldv,
    		    sort_svd, compute_uv, tol, max_sweeps);
            break;
            
        case torch::ScalarType::ComplexDouble:
            _batch_flattened_svd_cuda_helper<c10::complex<double>, double>(
                A.data_ptr<c10::complex<double>>(),
                d_S.data_ptr<double>(),
                d_U.data_ptr<c10::complex<double>>(),
                d_V.data_ptr<c10::complex<double>>(),
                info.data_ptr<int>(), batch_size, m, n, lda, ldu, ldv, 
    		    sort_svd, compute_uv, tol, max_sweeps);
            break;

        case torch::ScalarType::ComplexFloat:
            _batch_flattened_svd_cuda_helper<c10::complex<float>, float>(
                A.data_ptr<c10::complex<float>>(),
                d_S.data_ptr<float>(),
                d_U.data_ptr<c10::complex<float>>(),
                d_V.data_ptr<c10::complex<float>>(),
                info.data_ptr<int>(), batch_size, m, n, lda, ldu, ldv, 
    		    sort_svd, compute_uv, tol, max_sweeps);
            break;

        default:
            AT_ERROR("This function doesn't support types other than "
                     "float, double, complexFloat, and complexDouble.");
    }

    // Check error status
    if (info.ne(0).any().item().toInt()){
        TORCH_WARN("CUSolver (gesvdj) did not converge");
    }

    return std::make_tuple(d_U, d_S, d_V);
}

