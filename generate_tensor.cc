#include "generate_tensor.h"
#include "bert_types.h"
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/core/TensorImpl.h>
#include <cmath>
#include <cstdint>

namespace bert {

generate_tensor::generate_tensor() {

  // set the names
  impl_name = "generate_tensor_cpu";
  ud_name = "generate_tensor";

  // the following is the order of the tensors x mask query key value
  // multi_head_out_weights w1 w2
  inputTypes = {};

  // the output is a single tensor
  outputTypes = {"bert_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &generate_tensor::exec;
}

size_t generate_tensor::get_complexity_hint(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if (params.num_parameters() < 2) {
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void generate_tensor::get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in,
    bbts::ud_impl_t::meta_args_t &_out) const {

  // get the output argeters
  const bool has_bias = (int64_t)params.get_bool_or_default<0>(false);
  const std::uint32_t num_dims = (int64_t)params.get_int<1>();
  const std::uint32_t dim0 = (int64_t)params.get_int_or_default<2>(1);
  const std::uint32_t dim1 = (int64_t)params.get_int_or_default<3>(1);
  const std::uint32_t dim2 = (int64_t)params.get_int_or_default<4>(1);

  // get the output tensor
  auto &out = _out.get<0>().as<bert_dense_meta_t>().m();
  out = {.num_dim = num_dims,
         .dim0 = dim0,
         .dim1 = dim1,
         .dim2 = dim2,
         .has_bias = has_bias};
}

void generate_tensor::exec(const bbts::ud_impl_t::tensor_params_t &params,
                           const bbts::ud_impl_t::tensor_args_t &_in,
                           bbts::ud_impl_t::tensor_args_t &_out) {

  const bool has_bias = (int64_t)params.get_bool_or_default<0>(false);
  const std::uint32_t num_dims = (int64_t)params.get_int<1>();
  const std::uint32_t dim0 = (int64_t)params.get_int_or_default<2>(1);
  const std::uint32_t dim1 = (int64_t)params.get_int_or_default<3>(1);
  const std::uint32_t dim2 = (int64_t)params.get_int_or_default<4>(1);

  // get the output tensor
  auto &out = _out.get<0>().as<bert_dense_t>();
  auto &m_out = out.meta().m();
  m_out = {.num_dim = num_dims,
           .dim0 = dim0,
           .dim1 = dim1,
           .dim2 = dim2,
           .has_bias = has_bias};

  // set the new meta data
  at::Tensor out_tens = at::from_blob(out.data(), {dim0, dim1, dim2});
  rand_out(out_tens, {dim0, dim1, dim2});
}

} // namespace bert
