#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
at::Tensor group_transpose_cuda(at::Tensor &input, at::Tensor &coord_table,
                                at::Tensor &occurrences, at::Tensor &coord_ind);


// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


using namespace std;
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
    at::Tensor offset = torch::arange(0, shape[0], 
        torch::dtype(torch::kInt64).device(idx.device())).view(
        {shape[0], 1, 1});
    idx += offset*numPatches;

    idx = idx.flatten();
    // We sort the coordinates and keep their sorting order.
    at::Tensor coord_table = std::get<1>(idx.sort(0));
    int64_t bins = numPatches*shape[0];
    // Compute how many times each single-index coordinate appears in idx.
    at::Tensor occurrences = idx.bincount({}, bins).type_as(idx);
    at::Tensor coord_ind = torch::cumsum(occurrences,/* dim = */ 0, torch::kInt64);
    coord_ind = torch::cat({torch::zeros({1},
        torch::dtype(torch::kInt64).device(coord_ind.device())), 
        coord_ind.slice(0, 0, coord_ind.numel()-1)});
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



at::Tensor group_transpose(at::Tensor &input, at::Tensor &knn_idx) {
    const std::string err_ID = "group_transpose::Invalid Input: ";
    const std::string err_msg_args = "Input args #1 and #2 are expected to be a 4D and 3D-tensor, respectively.";
    
    CHECK_INPUT(input);
    CHECK_INPUT(knn_idx);    
    TORCH_CHECK(input.dim() == 4 and knn_idx.dim() == 3, err_ID, err_msg_args);
    // hvec = {coord_table, occurrences, coord_ind};
    std::vector<at::Tensor> hvec = group_transpose_helper(knn_idx);
    return group_transpose_cuda(input, hvec[0], hvec[1], hvec[2]);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("group_transpose", &group_transpose, "group transpose");
}