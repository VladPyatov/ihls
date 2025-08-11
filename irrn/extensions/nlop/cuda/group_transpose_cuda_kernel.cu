#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string.h>
#include <cstdint>


#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);

    }
}

inline int64_t divideUpwards(int64_t a, int64_t b)
{
    return (a + b - 1) / b ;
}


__device__ void get_indices(const int64_t *dims, const int64_t index, int64_t *ind){
    // index specifies one element of the 3D tensor X, i.e, index = b*F*P + f*P + p
    // where b = 0:B-1, f = 0:F-1, p = 0:P-1.

    // From the index = b*F*P + f*P + p  we want to recover :
    // ind[0] = b; ind[1] = f; ind[2]= p using dims[2] = {P, F}

    int64_t k = index / dims[0]; // k = b*F + f
    ind[0] = k / dims[1]; // ind[0] = b
    ind[1] = k % dims[1]; // ind[1] = f
    ind[2] = index % dims[0]; // ind[2] = p
}

/* Let us assume that X is of size B x F x P and I is of size
 * B x Nb x P (computed by misc.patchMatch) where
 * the elements of I have values in the range [0, P-1]. Then we can
 * create Y of size B x F x Nb x P by grouping together the similar
 * patches created according to the indexing in I.
 *
 * The transpose of this grouping operation can be computed as follows:
 *
 * If iz = b*F*P + f*P + p (iz = 0:B*F*P-1, Z[iz] = Z[b][f][p]),
 * then r = b*P + p (r = 0:B*P -1) and
 *
 *        coord_ind[r]+N[r]-1
 * Z[iz] =  Sum  Y[g(coord_table[k], iz)]
 *        k = coord_ind[r]
 *
 * N[r] = #occurrences of the single index r inside the neighbors-table I.
 * coord_table : Is an 1D-tensor of size B*Nb*H*W whose elements are all unique and lie in the range 0:B*Nb*P-1.
 * This tensor consists of B*P clusters of variable size. The size of each cluster (N[r]) is contained in the 1D
 * tensor occurrences.
 * coord_ind : Is an 1D-tensor of size B*H*W and indicates the first index of the cluster in coord_table.
 *
 * Now, let us assume that coord_table[k] = iv = b*Nb*P + nb*P + p1 and iz = b*F*P + f*P + p2. Then:
 * g(coord_table[k], iz) = g(iv, iz)  = ix' = b*F*Nb*P + f*Nb*P + nb*P + p1
 *
 */

template <typename scalar_t>
__global__ void GroupTranspose_cuda_kernel(const scalar_t *X, scalar_t *Y, const int64_t *coord_table,
        const int64_t *coord_ind, const int64_t *occurrences, const int64_t numel, const int64_t *dims){

    int64_t index = static_cast<int64_t>(threadIdx.x + blockIdx.x * blockDim.x);

    if (index < numel) {
        int64_t ind[3]; // {b, f, p}
        int64_t coord, coord_X;
        get_indices(dims, index, ind);

        // dims = {P, F, Nb}
        // Compute the index of occurrences and coord_ind which correspond to the output Y[index]
        int64_t index_t = ind[0]*dims[0] + ind[2]; // index_t = b*P + p

        for (int64_t k = 0; k < occurrences[index_t]; ++k) {
            // coord = b*Nb*P + nb*P + p = (b*Nb+nb)*P + p
            coord = coord_table[coord_ind[index_t] + k];
            int64_t i1 = coord % (dims[0]*dims[2]); // nb*P + p
            int64_t i2 = coord / (dims[0]*dims[2]); // b
            // coord_X = b*F*Nb*P + f*Nb*P + nb*P + p = ((b*F+f)*Nb+nb)*P + p
            coord_X = (i2*dims[1]+ind[1])*dims[0]*dims[2] + i1;
            Y[index] += X[coord_X];
        }
    }
}

template <typename scalar_t>
static inline cudaError_t GroupTranspose_cuda(const scalar_t *X, scalar_t *Y, const int64_t *coord_table,
                                      const int64_t *coord_ind, const int64_t *occurrences, const int64_t numel,
                                      const int64_t *dims){
    GroupTranspose_cuda_kernel<scalar_t>
            <<< divideUpwards(numel, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
            (X, Y, coord_table, coord_ind, occurrences, numel, dims);

    return cudaPeekAtLastError();
}

at::Tensor group_transpose_cuda(at::Tensor &input, at::Tensor &coord_table, at::Tensor &coord_ind,
        at::Tensor &occurrences) {
    const std::string err_ID = "group_transpose::Invalid Input: ";
    const std::string err_msg_dims = "Inputs dimensions mismatch.";
    const std::string err_msg_args = "Input argument #1 is expected to be a 4D-tensor.";

    // input must be a 3D tensor Batch x Features x Patches
    TORCH_CHECK(input.dim() == 4, err_ID, err_msg_args);

    //c10::IntList input_dims = input.sizes();
    const int64_t B = input.size(0);
    const int64_t F = input.size(1);
    const int64_t Nb = input.size(2);
    const int64_t P = input.size(3);

    // knn_index is a 1D tensor Batch * Nb (neighbors) * Patches
    TORCH_CHECK(coord_table.numel() / (B*P) == Nb, err_ID, err_msg_dims);

    int64_t dims[3] = {P, F, Nb};
    int64_t *dims_device;
    // cudaStat = cudaMalloc(&dims_device, 3*sizeof(int64_t));
    // assert(cudaStat == cudaSuccess);
    cudaMalloc(&dims_device, 3*sizeof(int64_t));
    cudaMemcpy(dims_device, dims, 3*sizeof(int64_t), cudaMemcpyHostToDevice);


    at::Tensor output = torch::zeros({B, F, P}, input.options());
    // at::Tensor output = input.new_zeros({B, F, P});
    switch (input.scalar_type()){
        case torch::ScalarType::Double:
            gpuErrchk(GroupTranspose_cuda<double>(input.data_ptr<double>(), output.data_ptr<double>(),
                    coord_table.data_ptr<int64_t>(), coord_ind.data_ptr<int64_t>(), occurrences.data_ptr<int64_t>(),
                    output.numel(), dims_device));
            break;
        case torch::ScalarType::Float:
            gpuErrchk(GroupTranspose_cuda<float>(input.data_ptr<float>(), output.data_ptr<float>(),
                    coord_table.data_ptr<int64_t>(), coord_ind.data_ptr<int64_t>(), occurrences.data_ptr<int64_t>(),
                    output.numel(), dims_device));
            break;
        case torch::ScalarType::ComplexDouble:
            GroupTranspose_cuda<c10::complex<double>>(input.data_ptr<c10::complex<double>>(),
                    output.data_ptr<c10::complex<double>>(), coord_table.data_ptr<int64_t>(),
                    coord_ind.data_ptr<int64_t>(), occurrences.data_ptr<int64_t>(),
                    output.numel(), dims_device);
            break;
        case torch::ScalarType::ComplexFloat:
            GroupTranspose_cuda<c10::complex<float>>(input.data_ptr<c10::complex<float>>(),
                    output.data_ptr<c10::complex<float>>(), coord_table.data_ptr<int64_t>(),
                    coord_ind.data_ptr<int64_t>(), occurrences.data_ptr<int64_t>(),
                    output.numel(), dims_device);
            break;
        default:
            TORCH_CHECK(false, "This function doesn't support any other types ",
                     "than float, double, complexFloat, and complexDouble.");
    }

    cudaFree(dims_device);
    return output;
}
