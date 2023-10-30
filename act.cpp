#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> cuda_logsumexp_pot_act_forward(
    const torch::Tensor x, const torch::Tensor weight, const torch::Tensor mu,
    const torch::Tensor sigma);

std::vector<torch::Tensor> cuda_logsumexp_pot_act_backward(
    const torch::Tensor x, const torch::Tensor weight,
    const torch::Tensor grad_out, const torch::Tensor grad_out_prime,
    const torch::Tensor mu, const torch::Tensor sigma,
    const torch::Tensor max_exp);

// C++ interface
#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)

std::vector<torch::Tensor>
logsumexp_pot_act_forward(const torch::Tensor x, const torch::Tensor weight,
                          const torch::Tensor mu, const torch::Tensor sigma) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(sigma);
  CHECK_INPUT(mu);

  return cuda_logsumexp_pot_act_forward(x, weight, mu, sigma);
}

std::vector<torch::Tensor>
logsumexp_pot_act_backward(const torch::Tensor x, const torch::Tensor weight,
                           const torch::Tensor f_grad,
                           const torch::Tensor f_grad_prime,
                           const torch::Tensor mu, const torch::Tensor sigma,
                           const torch::Tensor max_exp) {
  CHECK_INPUT(x);
  CHECK_INPUT(weight);
  CHECK_INPUT(f_grad);
  CHECK_INPUT(f_grad_prime);
  CHECK_INPUT(sigma);
  CHECK_INPUT(mu);

  return cuda_logsumexp_pot_act_backward(x, weight, f_grad, f_grad_prime, mu,
                                         sigma, max_exp);
}

// python interface
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lse_forward", &logsumexp_pot_act_forward,
        "logsumexp potential activation forward");
  m.def("lse_backward", &logsumexp_pot_act_backward,
        "logsumexp potential activation backward");
}
