#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/core/TensorImpl.h>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "bert_types.h"
#include "third_party/bbts/src/tensor/tensor.h"
#include "transformer.h"

namespace bert {

at::Tensor layer_norm(at::Tensor &x) {
  auto mean = x.mean(-1, true);
  auto std = x.std(-1, true, true);
  return (x - mean) / (std + 1e-6);
}

at::Tensor attention(at::Tensor &query, at::Tensor &key, at::Tensor &value,
                     at::Tensor *mask) {
  auto scores =
      at::matmul(query, key.transpose(-2, -1)) / std::sqrt(query.size(-1));
  if (mask == nullptr) {
    scores = scores.masked_fill(*mask == 0, -1e9);
  }
  auto p_attn = at::softmax(scores, -1);
  return at::matmul(p_attn, value);
}

at::Tensor multihead_attention(const std::int64_t batch_size,
                               const std::int64_t seq_len,
                               const std::int64_t num_heads,
                               const std::int64_t hidden_layer_size,
                               at::Tensor &q_w, at::Tensor &q_b,
                               at::Tensor &k_w, at::Tensor &k_b,
                               at::Tensor &v_w, at::Tensor &v_b, at::Tensor &x,
                               at::Tensor &multihead_out_w,
                               at::Tensor &multihead_out_b, at::Tensor *mask) {

  auto q_r = at::matmul(x, q_w.transpose(0, 1)) + q_b;
  auto k_r = at::matmul(x, k_w.transpose(0, 1)) + k_b;
  auto v_r = at::matmul(x, v_w.transpose(0, 1)) + v_b;

  q_r = q_r.view({batch_size, -1, num_heads, hidden_layer_size / num_heads})
            .transpose(1, 2);
  k_r = k_r.view({batch_size, -1, num_heads, hidden_layer_size / num_heads})
            .transpose(1, 2);
  v_r = v_r.view({batch_size, -1, num_heads, hidden_layer_size / num_heads})
            .transpose(1, 2);

  x = attention(q_r, k_r, v_r, mask);
  x = x.transpose(1, 2).contiguous().view({batch_size, -1, hidden_layer_size});
  return at::matmul(x, multihead_out_w.transpose(0, 1)) + multihead_out_b;
}

void pointwise_ffnn(at::Tensor &w1_w, at::Tensor &w1_b, at::Tensor &w2_w,
                    at::Tensor &w2_b, at::Tensor &x, at::Tensor &out) {
  x = at::gelu(at::matmul(x, w1_w.transpose(0, 1)) + w1_b);
  at::matmul_out(out, x, w2_w.transpose(0, 1)) + w2_b;
}

void transformer(const std::int64_t batch_size, const std::int64_t seq_len,
                 const std::int64_t num_heads,
                 const std::int64_t hidden_layer_size, at::Tensor &w1_w,
                 at::Tensor &w1_b, at::Tensor &w2_w, at::Tensor &w2_b,
                 at::Tensor &q_w, at::Tensor &q_b, at::Tensor &k_w,
                 at::Tensor &k_b, at::Tensor &v_w, at::Tensor &v_b,
                 at::Tensor &multihead_out_w, at::Tensor &multihead_out_b,
                 at::Tensor &x, at::Tensor *mask, at::Tensor &out) {

  // apply the layer norm
  x = layer_norm(x);

  // residual connection after the multihead attention
  x += multihead_attention(batch_size, seq_len, num_heads, hidden_layer_size,
                           q_w, q_b, k_w, k_b, v_w, v_b, x, multihead_out_w,
                           multihead_out_b, mask);

  // apply the layer normalization
  x = layer_norm(x);

  // the final output of the transformer
  pointwise_ffnn(w1_w, w1_b, w2_w, w2_b, x, out);
}

void kernel(const std::int64_t batch_size, const std::int64_t seq_len,
            const std::int64_t num_heads, const std::int64_t hidden_layer_size,
            bert_dense_t &_x, bert_dense_t &_mask, bert_dense_t &_q,
            bert_dense_t &_k, bert_dense_t &_v, bert_dense_t &_multihead_out,
            bert_dense_t &_w1, bert_dense_t &_w2, bert_dense_t &_out) {

  at::Tensor x =
      at::from_blob(_x.data(), {batch_size, seq_len, hidden_layer_size});
  at::Tensor mask =
      at::from_blob(_mask.data(), {batch_size, 1, seq_len, seq_len});

  at::Tensor q_w =
      at::from_blob(_q.data(), {hidden_layer_size, hidden_layer_size});
  at::Tensor q_b = at::from_blob(_q.bias(), {1, hidden_layer_size});

  at::Tensor k_w =
      at::from_blob(_k.data(), {hidden_layer_size, hidden_layer_size});
  at::Tensor k_b = at::from_blob(_k.bias(), {1, hidden_layer_size});

  at::Tensor v_w =
      at::from_blob(_v.data(), {hidden_layer_size, hidden_layer_size});
  at::Tensor v_b = at::from_blob(_v.bias(), {1, hidden_layer_size});

  at::Tensor multihead_out_w = at::from_blob(
      _multihead_out.data(), {hidden_layer_size, hidden_layer_size});
  at::Tensor multihead_out_b =
      at::from_blob(_multihead_out.bias(), {1, hidden_layer_size});

  at::Tensor w1_w =
      at::from_blob(_w1.data(), {4 * hidden_layer_size, hidden_layer_size});
  at::Tensor w1_b = at::from_blob(_w1.bias(), {1, 4 * hidden_layer_size});

  at::Tensor w2_w =
      at::from_blob(_w2.data(), {hidden_layer_size, 4 * hidden_layer_size});
  at::Tensor w2_b = at::from_blob(_w2.bias(), {1, hidden_layer_size});

  at::Tensor out =
      at::from_blob(_out.data(), {batch_size, seq_len, hidden_layer_size});

  transformer(batch_size, seq_len, num_heads, hidden_layer_size, w1_w, w1_b,
              w2_w, w2_b, q_w, q_b, k_w, k_b, v_w, v_b, multihead_out_w,
              multihead_out_b, x, &mask, out);
}

bert::transformer::transformer() {

  // set the names
  impl_name = "transformer_cpu";
  ud_name = "transformer";

  // the following is the order of the tensors x mask query key value
  // multi_head_out_weights w1 w2
  inputTypes = {"bert_dense", "bert_dense", "bert_dense", "bert_dense",
                "bert_dense", "bert_dense", "bert_dense", "bert_dense"};

  // the output is a single tensor
  outputTypes = {"bert_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &transformer::exec;
}

size_t bert::transformer::get_complexity_hint(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if (params.num_parameters() < 4) {
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bert::transformer::get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in,
    bbts::ud_impl_t::meta_args_t &_out) const {

  const std::uint32_t batch_size = params.get_int<0>();
  const std::uint32_t seq_len = params.get_int<1>();
  const std::uint32_t num_heads = params.get_int<2>();
  const std::uint32_t hidden_layer_size = params.get_int<3>();

  // get the output argeters
  auto &m_out = _out.get<0>().as<bert_dense_meta_t>().m();

  // set the new values
  m_out = {.num_dim = 3,
           .dim0 = batch_size,
           .dim1 = seq_len,
           .dim2 = hidden_layer_size};
}

void bert::transformer::exec(const bbts::ud_impl_t::tensor_params_t &params,
                             const bbts::ud_impl_t::tensor_args_t &_in,
                             bbts::ud_impl_t::tensor_args_t &_out) {

  const std::uint32_t batch_size = params.get_int<0>();
  const std::uint32_t seq_len = params.get_int<1>();
  const std::uint32_t num_heads = params.get_int<2>();
  const std::uint32_t hidden_layer_size = params.get_int<3>();

  // get the output tensor
  auto &out = _out.get<0>().as<bert_dense_t>();

  // get the input tensors
  auto &x = _in.get<0>().as<bert_dense_t>();
  auto &mask = _in.get<0>().as<bert_dense_t>();
  auto &q = _in.get<0>().as<bert_dense_t>();
  auto &k = _in.get<0>().as<bert_dense_t>();
  auto &v = _in.get<0>().as<bert_dense_t>();
  auto &multihead_out = _in.get<0>().as<bert_dense_t>();
  auto &w1 = _in.get<0>().as<bert_dense_t>();
  auto &w2 = _in.get<0>().as<bert_dense_t>();

  kernel(batch_size, seq_len, num_heads, hidden_layer_size, x, mask, q, k, v,
         multihead_out, w1, w2, out);
}

} // namespace bert
