#include "fbmd_cuda.h"


std::tuple<at::Tensor, at::Tensor> batch_symeig_cuda(
    at::Tensor &mat, bool eigenvectors=true, bool upper=true, double tol=1e-7, int max_sweeps=15) {
  
  CHECK_CUDA(mat)
  CHECK_TYPE(mat)
  CHECK_CONTIGUOUS(mat)
  CHECK_SQUARE(mat)
  CHECK_DIMS(mat)

//  int max_sweeps = 15;
  int sort_eig = 1; // sort the eigenvalues in ascending order

  auto shape = mat.sizes();
  auto batch_size = batchCount(mat);
  at::Tensor flattened_mat = mat.clone().reshape(
      {batch_size, mat.size(-2), mat.size(-1)}).contiguous();

  at::Tensor E;
  at::Tensor V;

  std::tie(E, V) = _batch_flattened_symeig_cuda(
      flattened_mat, sort_eig, eigenvectors, upper, tol, max_sweeps);

  E = E.reshape(shape.slice(0, mat.dim()-1)).contiguous();
  if (eigenvectors)
      V = V.reshape(shape).contiguous();
  else /* If eigenvectors=false we set V to an empty tensor */
      // V = torch::zeros_like(mat);
      V = torch::empty(0, mat.options());
  return std::make_tuple(E, V);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_svd_cuda(
    at::Tensor &mat, bool compute_uv=true, double tol=1e-7, int max_sweeps=15) {

    CHECK_CUDA(mat)
    CHECK_TYPE(mat)
    CHECK_CONTIGUOUS(mat)
    CHECK_DIMS(mat)

//    int max_sweeps = 15;
    int sort_svd = 1; // sort the singular values in descending order.

    auto shape = mat.sizes();
    auto m = mat.size(-2);
    auto n = mat.size(-1);
    auto min_mn = (m < n) ? m : n;
    
    auto batch_size = batchCount(mat);
    at::Tensor flattened_mat = 
        mat.clone().reshape({batch_size, mat.size(-2), mat.size(-1)}).contiguous();

    at::Tensor U;
    at::Tensor S;
    at::Tensor V;

    std::tie(U, S, V) = _batch_flattened_svd_cuda(
        flattened_mat, sort_svd, compute_uv, tol, max_sweeps);
    
    auto fshape = shape.slice(0, mat.dim()-2).vec();
    fshape.push_back(min_mn);
    S = S.reshape(fshape).contiguous();

    if (compute_uv){
        fshape.pop_back();
        fshape.push_back(n);
        fshape.push_back(min_mn);    
        V = V.transpose(-1, -2).slice(-1, 0, min_mn).reshape(fshape).contiguous();
    
        fshape.pop_back();   
        fshape.pop_back();
        fshape.push_back(m);
        fshape.push_back(min_mn);
        U = U.transpose(-1, -2).slice(-1, 0, min_mn).reshape(fshape).contiguous();
    }
    else{ /* If compute_uv=false we set U, V to empty tensors. */
        U = torch::empty(0, mat.options());
        V = torch::empty(0, mat.options());
        /* fshape.pop_back();
        fshape.push_back(n);
        fshape.push_back(min_mn);        
        V = torch::zeros(fshape, S.options());
        
        fshape.pop_back();   
        fshape.pop_back();
        fshape.push_back(m);
        fshape.push_back(min_mn);        
        U = torch::zeros(fshape, S.options());*/
    }
    
    return std::make_tuple(U, S, V);
}

at::Tensor F_approximation(const at::Tensor& E, int order, bool descending){
    auto K = E.unsqueeze(-1).div(E.unsqueeze(-2));
    auto R = at::zeros_like(K);
    for (int i=0; i < order+1; ++i)
        R = R + K.pow(i);

    R = R.mul(E.unsqueeze(-2).pow(-1));
    if (descending == true)
        R = at::tril(R);
    else
        R = at::triu(R);

    auto F = R - R.transpose(-1, -2);
    return F;
}
        
// Backward functions for svd and symeig copied from 
// https://github.com/pytorch/pytorch/blob/57dcb04239019f67a7cc72d9fd43018fd37d226c/torch/csrc/autograd/FunctionsManual.cpp

// http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf
at::Tensor batch_symeig_backward(const std::vector<at::Tensor> &grads, const at::Tensor& self,
    bool eigenvectors, bool upper, const at::Tensor& lambda, const at::Tensor& v) {
  // This gradient is symmetric, and not triangular.
  // symeig operates only on symmetric inputs, which is a subspace of
  // R^{n x n}, and hence the derivative is not well-defined for off-diagonal
  // elements. We resolve this by taking the gradient of the functionally independent
  // elements of the matrix (i.e., the lower triangular portion of the input) and then
  // reflect it on the upper triangular portion, thereby symmetrizing the gradient of
  // the symeig operation. The motivation behind this choice is that symmetric gradient
  // leads to stable gradient updates, and retains symmetry of the updated matrix if it
  // were updated by a gradient based algorithm.
  TORCH_CHECK(eigenvectors, 
      "symeig_backward: Setting eigenvectors to false in torch.symeig doesn't compute eigenvectors ",
      "and hence we cannot compute backward. Please use torch.symeig(eigenvectors=True)");

  auto glambda = grads[0];
  auto gv = grads[1];

  auto vt = v.transpose(-2, -1);

  at::Tensor result;
  if (gv.defined()) {
      at::Tensor F = lambda.unsqueeze(-2) - lambda.unsqueeze(-1);
      F.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(INFINITY);
      F.pow_(-1);
      F.mul_(at::matmul(vt, gv));
      result = at::matmul(v, at::matmul(F, vt));
  } else {
      result = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (glambda.defined()) {
      result.add_(at::matmul(at::matmul(v, at::diag_embed(glambda, /*offset=*/0, /*dim1=*/-2, /*dim2=*/-1)), vt));
  }
  return result.add(result.transpose(-2, -1)).mul_(0.5);
}

      
at::Tensor batch_robust_symeig_backward(const std::vector<at::Tensor> &grads,
    const at::Tensor& self, bool eigenvectors, bool upper,
    const at::Tensor& lambda, const at::Tensor& v, const int order) {
  // This gradient is symmetric, and not triangular.
  // symeig operates only on symmetric inputs, which is a subspace of
  // R^{n x n}, and hence the derivative is not well-defined for off-diagonal
  // elements. We resolve this by taking the gradient of the functionally independent
  // elements of the matrix (i.e., the lower triangular portion of the input) and then
  // reflect it on the upper triangular portion, thereby symmetrizing the gradient of
  // the symeig operation. The motivation behind this choice is that symmetric gradient
  // leads to stable gradient updates, and retains symmetry of the updated matrix if it
  // were updated by a gradient based algorithm.
  TORCH_CHECK(eigenvectors, 
      "robust_symeig_backward: Setting eigenvectors to false in torch.symeig doesn't compute eigenvectors ",
      "and hence we cannot compute backward. Please use torch.symeig(eigenvectors=True)");

  auto glambda = grads[0];
  auto gv = grads[1];

  auto vt = v.transpose(-2, -1);

  at::Tensor result;
  if (gv.defined()) {
      at::Tensor F = F_approximation(lambda, order, false);
      F.mul_(at::matmul(vt, gv));
      result = at::matmul(v, at::matmul(F, vt));
  } else {
      result = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (glambda.defined()) {
      result.add_(at::matmul(at::matmul(v, at::diag_embed(glambda, /*offset=*/0, /*dim1=*/-2, /*dim2=*/-1)), vt));
  }
  return result.add(result.transpose(-2, -1)).mul_(0.5);
}
      
// https://j-towns.github.io/papers/svd-derivative.pdf
//
// This makes no assumption on the signs of sigma.
at::Tensor batch_svd_backward(const std::vector<at::Tensor> &grads, const at::Tensor& self,
          bool some, bool compute_uv, const at::Tensor& raw_u,
          const at::Tensor& sigma, const at::Tensor& raw_v) {
  TORCH_CHECK(compute_uv,
           "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
           "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = sigma.size(-1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(-1, 0, k);
    v = raw_v.narrow(-1, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(-1, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(-1, 0, k);
    }
  }
  auto vt = v.transpose(-2, -1);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = at::matmul(u, at::matmul(gsigma.diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1), vt));
  } else {
    sigma_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(-2, -1);
  auto im = at::eye(m, self.options());
  auto in = at::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  auto sigma_mat_inv = sigma.pow(-1).diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  auto sigma_sq = sigma.pow(2);
  auto F = sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1);
  // The following two lines invert values of F, and fills the diagonal with 0s.
  // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
  // first to prevent nan from appearing in backward of this function.
  F.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(INFINITY);
  F = F.pow(-1);

  at::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term = at::matmul(u, at::matmul(F.mul(at::matmul(ut, gu) - at::matmul(gu.transpose(-2, -1), u)), sigma_mat));
    if (m > k) {
      u_term = u_term + at::matmul(im - at::matmul(u, ut), at::matmul(gu, sigma_mat_inv));
    }
    u_term = at::matmul(u_term, vt);
  } else {
    u_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(-2, -1);
    v_term = at::matmul(sigma_mat, at::matmul(F.mul(at::matmul(vt, gv) - at::matmul(gvt, v)), vt));
    if (n > k) {
      v_term = v_term + at::matmul(sigma_mat_inv, at::matmul(gvt, in - at::matmul(v, vt)));
    }
    v_term = at::matmul(u, v_term);
  } else {
    v_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return u_term + sigma_term + v_term;
}


at::Tensor batch_robust_svd_backward(const std::vector<at::Tensor> &grads,
          const at::Tensor& self, bool some, bool compute_uv,
          const at::Tensor& raw_u, const at::Tensor& sigma,
          const at::Tensor& raw_v, const int order) {
  TORCH_CHECK(compute_uv,
           "robust_svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
           "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = sigma.size(-1);
  auto gsigma = grads[1];

  auto u = raw_u;
  auto v = raw_v;
  auto gu = grads[0];
  auto gv = grads[2];

  if (!some) {
    // We ignore the free subspace here because possible base vectors cancel
    // each other, e.g., both -v and +v are valid base for a dimension.
    // Don't assume behavior of any particular implementation of svd.
    u = raw_u.narrow(-1, 0, k);
    v = raw_v.narrow(-1, 0, k);
    if (gu.defined()) {
      gu = gu.narrow(-1, 0, k);
    }
    if (gv.defined()) {
      gv = gv.narrow(-1, 0, k);
    }
  }
  auto vt = v.transpose(-2, -1);

  at::Tensor sigma_term;
  if (gsigma.defined()) {
    sigma_term = at::matmul(u, at::matmul(gsigma.diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1), vt));
  } else {
    sigma_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  // in case that there are no gu and gv, we can avoid the series of kernel
  // calls below
  if (!gv.defined() && !gu.defined()) {
    return sigma_term;
  }

  auto ut = u.transpose(-2, -1);
  auto im = at::eye(m, self.options());
  auto in = at::eye(n, self.options());
  auto sigma_mat = sigma.diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  auto sigma_mat_inv = sigma.pow(-1).diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
  auto sigma_sq = sigma.pow(2);
  auto F = F_approximation(sigma_sq, order, true);
  
  at::Tensor u_term, v_term;

  if (gu.defined()) {
    u_term = at::matmul(u, at::matmul(F.mul(at::matmul(ut, gu) - at::matmul(gu.transpose(-2, -1), u)), sigma_mat));
    if (m > k) {
      u_term = u_term + at::matmul(im - at::matmul(u, ut), at::matmul(gu, sigma_mat_inv));
    }
    u_term = at::matmul(u_term, vt);
  } else {
    u_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (gv.defined()) {
    auto gvt = gv.transpose(-2, -1);
    v_term = at::matmul(sigma_mat, at::matmul(F.mul(at::matmul(vt, gv) - at::matmul(gvt, v)), vt));
    if (n > k) {
      v_term = v_term + at::matmul(sigma_mat_inv, at::matmul(gvt, in - at::matmul(v, vt)));
    }
    v_term = at::matmul(u, v_term);
  } else {
    v_term = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return u_term + sigma_term + v_term;
}      

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("batch_svd_cuda", &batch_svd_cuda, "Batch svd solver (CUDA)");
  m.def("batch_svd_backward", &batch_svd_backward, "Batch svd backward");
  m.def("batch_robust_svd_backward", &batch_robust_svd_backward, "Batch robust svd backward");
  m.def("batch_symeig_cuda", &batch_symeig_cuda, "Batch symeig solver (CUDA)");
  m.def("batch_symeig_backward", &batch_symeig_backward, "Batch symeig backward");
  m.def("batch_robust_symeig_backward", &batch_robust_symeig_backward, "Batch robust symeig backward");
}

