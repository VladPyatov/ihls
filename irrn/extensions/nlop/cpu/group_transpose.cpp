#include <torch/extension.h>
#include <vector>
#include <omp.h>
#include <string.h>
#include <cstdint>

/* To access the data of a torch::Tensor
 * at::Tensor x = torch::Tensor randn({5,3,2},torch::kFloat64);
 * auto *x_ptr = x.data<double>();
 * auto x_accessor = x.accessor<double,x.dim()>();
 * int64_t i,j,z;
 * c10::IntList dims = x.sizes();
 * x_accessor[i][j][z] = x_ptr[i*dims[1]*dims[2]+j*dims[2]+z]
 *
 */

std::vector<at::Tensor> group_transpose_helper(at::Tensor &knn_idx){
    // knn_idx is a 3D tensor of size B x Nb x P, where B: number of batches, Nb: number of neighbors and
    // P:number of patches.
    at::Tensor idx = knn_idx.clone();

    auto shape = idx.sizes();
    int64_t numPatches = shape[2];
    //int64_t *data_ptr = nbr_indices.data<int64_t>();

    // idx initially contains unique elements in the range 0:P-1.
    // In order to differentiate between the coordinates of patches in different
    // images in a batch we need to transform the coordinates to the form:
    // l = b*P + p where b = 0:B-1 and p = 0:P-1
    at::Tensor offset = torch::arange(0, shape[0], torch::kInt64).view({shape[0], 1, 1});
    idx += offset*numPatches;

    idx = idx.flatten();
    // We sort the coordinates and keep their sorting order.
    at::Tensor coord_table = std::get<1>(idx.sort(0));
    int64_t bins = numPatches*shape[0];
    // Compute how many times each single-index coordinate appears in idx.
    at::Tensor occurrences = idx.bincount({}, bins).type_as(idx);
    at::Tensor coord_ind = torch::cumsum(occurrences,/* dim = */ 0, torch::kInt64);
    coord_ind = torch::cat({torch::zeros({1}, torch::kInt64), coord_ind.slice(0, 0, coord_ind.numel()-1)});

    // Let us indicate Z = patchGroup_transpose(Y, idx).
    // Then, the element with index = b*P + p in occurrences indicates how many elements in Y need to be
    // included in the sum that will produce the value for the tensor element Z[b][:][p].

    // coord_ind[index] points to the first position in coord_table which holds the coordinates of the elements
    // of Y that contribute to the output of Z[b][:][p]. That is
    // coord_ind[index] : coord_ind[index] + occurrences[index] holds the coordinates of the elements in Y that
    // contribute to the result of Z[b][:][p].

    // coord_table consists of unique indices in the range [0, B*Nb*P-1]. We use the single index idx = (b*Nb+nb)*P + p
    // to extract the spatial positions (b, p) of Y output of Z[b][:][p].
    return {coord_table.contiguous(), coord_ind.contiguous(), occurrences.contiguous()};
}

void get_indices(const int64_t *dims, const int64_t index, int64_t *ind){
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
void GroupTranspose_(const scalar_t *X, scalar_t *Y, const int64_t *coord_table,
                     const int64_t *coord_ind, const int64_t *occurrences,
                     const int64_t index, const int64_t *dims){

    // index's value is in the range 0:B*F*P-1
    int64_t *ind = new int64_t[3]; // {b, f, p}
    int64_t coord, coord_X;
    get_indices(dims, index, ind);

    // dims = {P, F, Nb}
    // Compute the index of occurrences and coord_ind which correspond to the output Y[index]
    int64_t index_t = ind[0]*dims[0]+ind[2]; // index_t = b*P + p

    for(int64_t k = 0; k < occurrences[index_t]; ++k){
        // coord = b*Nb*P + nb*P + p = (b*Nb+nb)*P + p
        coord = coord_table[coord_ind[index_t]+k]; 
        int64_t i1 = coord % (dims[0]*dims[2]); // nb*P + p
        int64_t i2 = coord / (dims[0]*dims[2]); // b
        // coord_X = b*F*Nb*P + f*Nb*P + nb*P + p = ((b*F+f)*Nb+nb)*P + p
        coord_X = (i2*dims[1]+ind[1])*dims[0]*dims[2] + i1;
        Y[index] += X[coord_X];
    }

    delete[] ind;
}


template <typename scalar_t>
void GroupTranspose(const scalar_t *X, scalar_t *Y, const int64_t *coord_table,
            const int64_t *coord_ind, const int64_t *occurrences,
            const int64_t numel, const int64_t *dims){
    int64_t elem;
    #pragma omp parallel for private(elem)
    for (elem = 0; elem < numel; ++elem)
        GroupTranspose_<scalar_t>(X, Y, coord_table, coord_ind, occurrences, elem, dims);
}


at::Tensor group_transpose(at::Tensor &input, at::Tensor &knn_idx) {
    const std::string err_ID = "group_transpose::Invalid Input: ";
    const std::string err_msg_type = "Input argument #2 must be of Long type.";
    const std::string err_msg_dims = "Inputs dimensions mismatch.";
    const std::string err_msg_args = "Input args #1 and #2 are expected to be 4D and 3D-tensors, respectively.";
    const std::string err_msg_contiguous = "All input arguments must be contiguous.";

    TORCH_CHECK(input.is_contiguous() and knn_idx.is_contiguous(), err_ID, err_msg_contiguous);
    // knn indices must be of Long type.
    TORCH_CHECK(knn_idx.dtype() == torch::kInt64, err_ID, err_msg_type);

    // input must be a 3D tensor Batch x Features x Patches and knn_index a 3D tensor Batch x Nb x Patches
    TORCH_CHECK(input.dim() == 4 and knn_idx.dim() == 3, err_ID, err_msg_args);

    //c10::IntList input_dims = input.sizes();
    auto B = input.size(0);
    auto F = input.size(1);
    auto Nb = input.size(2);
    auto P = input.size(3);

    // knn_index is a 3D tensor Batch x Nb (neighbors) x Patches
    TORCH_CHECK(knn_idx.size(0) == B, err_ID, err_msg_dims);
    TORCH_CHECK(knn_idx.size(1) == Nb, err_ID, err_msg_dims);
    TORCH_CHECK(knn_idx.size(2) == P, err_ID, err_msg_dims);

    auto *dims = new int64_t[3] {P, F, Nb};
    /*
    const auto *nbr_indices_ptr = nbr_indices.data<int64_t>();
    const auto *input_ptr = input.data<scalar_t>();
    const auto *weights_ptr = weights.data<scalar_t>();
    const int64_t *dims_ptr = dims.data();
     */
    at::Tensor output = input.new_zeros({B, F, P});
    // hvec = {coord_table, occurrences, coord_ind}
    std::vector<at::Tensor> hvec = group_transpose_helper(knn_idx);

    switch (input.scalar_type()){
        case torch::ScalarType::Double:
            GroupTranspose<double>(input.data_ptr<double>(), output.data_ptr<double>(), hvec[0].data_ptr<int64_t>(),
                    hvec[1].data_ptr<int64_t>(), hvec[2].data_ptr<int64_t>(), output.numel(), dims);
            break;
        case torch::ScalarType::Float:
            GroupTranspose<float>(input.data_ptr<float>(), output.data_ptr<float>(), hvec[0].data_ptr<int64_t>(),
                                   hvec[1].data_ptr<int64_t>(), hvec[2].data_ptr<int64_t>(), output.numel(), dims);
            break;
        case torch::ScalarType::ComplexDouble:
            GroupTranspose<c10::complex<double>>(input.data_ptr<c10::complex<double>>(),
                    output.data_ptr<c10::complex<double>>(), hvec[0].data_ptr<int64_t>(), hvec[1].data_ptr<int64_t>(),
                    hvec[2].data_ptr<int64_t>(), output.numel(), dims);
            break;
        case torch::ScalarType::ComplexFloat:
            GroupTranspose<c10::complex<float>>(input.data_ptr<c10::complex<float>>(),
                    output.data_ptr<c10::complex<float>>(), hvec[0].data_ptr<int64_t>(), hvec[1].data_ptr<int64_t>(),
                    hvec[2].data_ptr<int64_t>(), output.numel(), dims);
            break;
        default:
            TORCH_CHECK(false, "This function doesn't support any other types than float, "
                     "double, complexFloat, and complexDouble.");
    }

    delete[] dims;
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("group_transpose", &group_transpose, "group_transpose");
}