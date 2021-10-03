#include "bert_types.h"
#include "transformer.h"
#include <memory>

using namespace bert;

int main() {

  const std::int64_t batch_size = 1;
  const std::int64_t seq_len = 20;
  const std::int64_t num_heads = 8;
  const std::int64_t hidden_layer_size = 256;

  auto fns = bert_dense_t::get_creation_fs();

  bert_dense_meta_t x_meta(0, false, batch_size, seq_len, hidden_layer_size);
  bert_dense_meta_t mask_meta(0, false, batch_size, seq_len, seq_len);

  bert_dense_meta_t q_meta(0, true, hidden_layer_size, hidden_layer_size);
  bert_dense_meta_t k_meta(0, true, hidden_layer_size, hidden_layer_size);
  bert_dense_meta_t v_meta(0, true, hidden_layer_size, hidden_layer_size);

  bert_dense_meta_t multihead_out_meta(0, true, hidden_layer_size,
                                       hidden_layer_size);
  bert_dense_meta_t w1_meta(0, true, 4 * hidden_layer_size, hidden_layer_size);
  bert_dense_meta_t w2_meta(0, true, hidden_layer_size, 4 * hidden_layer_size);
  bert_dense_meta_t out_meta(0, false, batch_size, seq_len, hidden_layer_size);

  std::unique_ptr<char[]> x_mem(new char[fns.get_size(x_meta)]);
  std::unique_ptr<char[]> mask_mem(new char[fns.get_size(mask_meta)]);
  std::unique_ptr<char[]> q_mem(new char[fns.get_size(q_meta)]);
  std::unique_ptr<char[]> k_mem(new char[fns.get_size(k_meta)]);
  std::unique_ptr<char[]> v_mem(new char[fns.get_size(v_meta)]);
  std::unique_ptr<char[]> multihead_out_mem(
      new char[fns.get_size(multihead_out_meta)]);
  std::unique_ptr<char[]> w1_mem(new char[fns.get_size(w1_meta)]);
  std::unique_ptr<char[]> w2_mem(new char[fns.get_size(w2_meta)]);
  std::unique_ptr<char[]> out_mem(new char[fns.get_size(out_meta)]);

  auto &x = fns.init_tensor(x_mem.get(), x_meta).as<bert_dense_t>();
  auto &mask = fns.init_tensor(mask_mem.get(), mask_meta).as<bert_dense_t>();
  auto &q = fns.init_tensor(q_mem.get(), q_meta).as<bert_dense_t>();
  auto &k = fns.init_tensor(k_mem.get(), k_meta).as<bert_dense_t>();
  auto &v = fns.init_tensor(v_mem.get(), v_meta).as<bert_dense_t>();
  auto &multihead_out =
      fns.init_tensor(multihead_out_mem.get(), multihead_out_meta)
          .as<bert_dense_t>();
  auto &w1 = fns.init_tensor(w1_mem.get(), w1_meta).as<bert_dense_t>();
  auto &w2 = fns.init_tensor(w2_mem.get(), w2_meta).as<bert_dense_t>();
  auto &out = fns.init_tensor(out_mem.get(), out_meta).as<bert_dense_t>();

  std::vector<bbts::command_param_t> _params = {
      bbts::command_param_t{.i = batch_size},
      bbts::command_param_t{.i = seq_len},
      bbts::command_param_t{.i = num_heads},
      bbts::command_param_t{.i = hidden_layer_size},
  };
  auto params =
      bbts::command_param_list_t{._data = _params.data(), ._num_elements = 0};
  bbts::ud_impl_t::tensor_args_t input_args = {
      {&x, &mask, &q, &k, &v, &multihead_out, &w1, &w2}};
  bbts::ud_impl_t::tensor_args_t output_args = {{&out}};

  transformer fun;
  fun.exec(bbts::ud_impl_t::tensor_params_t{params}, input_args, output_args);

  return 0;
}