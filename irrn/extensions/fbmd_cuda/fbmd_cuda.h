#include <torch/extension.h>

static inline int64_t batchCount(const at::Tensor& batched_matrices) {
  int64_t result = 1;
  for (int64_t i = 0; i < batched_matrices.ndimension() - 2; i++) {
    result *= batched_matrices.size(i);
  }
  return result;
}

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x                \
                                        " must be contiguous")
                                        
#define CHECK_SQUARE(x) TORCH_CHECK(x.size(-2) == x.size(-1), #x             \
                                    " must be square")
                                    
#define IS_FLOAT(x) (x.scalar_type() == torch::ScalarType::Float)

#define IS_DOUBLE(x) (x.scalar_type() == torch::ScalarType::Double)

#define IS_COMPLEXFLOAT(x) (x.scalar_type() == torch::ScalarType::ComplexFloat)

#define IS_COMPLEXDOUBLE(x) (x.scalar_type() ==                              \
                             torch::ScalarType::ComplexDouble)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_TYPE(x) TORCH_CHECK(                                           \
    (IS_FLOAT(x)) || (IS_COMPLEXFLOAT(x)) || (IS_DOUBLE(x)) ||               \
    (IS_COMPLEXDOUBLE(x)), #x                                                \
    " must be either a Float, ComplexFloat, Double, or ComplexDouble tensor")

#define CHECK_DIMS(x) TORCH_CHECK(x.size(-2) <= 32 and x.size(-1) <= 32, #x  \
    " must have dimensions which are less or equal to 32.")

// Cuda function definitions
std::tuple<at::Tensor, at::Tensor> batch_symeig_cuda(
    at::Tensor &mat, bool eigenvectors, bool upper, double tol, int max_sweeps);

std::tuple<at::Tensor, at::Tensor> _batch_flattened_symeig_cuda(
    at::Tensor &flattened_mat, int sort_eig, bool eigenvectors, bool upper,
    double tol, int max_sweeps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_svd_cuda(
    at::Tensor &mat, bool compute_uv, double tol, int max_sweeps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> _batch_flattened_svd_cuda(
    at::Tensor &flattened_mat, int sort_svd, bool compute_uv, double tol,
    int max_sweeps);
