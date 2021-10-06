#include "third_party/bbts/src/commands/compile_source_file.h"
#include "third_party/bbts/src/commands/two_layer_compiler.h"
#include "third_party/bbts/src/tensor/tensor.h"
#include <cstdint>
#include <map>

using namespace bbts;

tid_t current_tid = 0;

const int32_t GENERATE_TENSOR_ID = 0;
const int32_t TRANSFORMER_ID = 1;

struct transformer_weights_t {
  tid_t q_tid;
  tid_t k_tid;
  tid_t v_tid;
  tid_t multihead_out_tid;
  tid_t w1_tid;
  tid_t w2_tid;
};

transformer_weights_t
generate_transformer_tensors(std::uint32_t batch_size, std::uint32_t seq_len,
                             std::uint32_t num_heads,
                             std::uint32_t hidden_layer_size,
                             std::vector<abstract_command_t> &commands) {
  transformer_weights_t out{};

  // bert_dense_meta_t q_meta(0, true, hidden_layer_size, hidden_layer_size);
  out.q_tid = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = true}, command_param_t{.u = 2},
                 command_param_t{.u = hidden_layer_size},
                 command_param_t{.u = hidden_layer_size}}});

  // bert_dense_meta_t k_meta(0, true, hidden_layer_size, hidden_layer_size);
  out.k_tid = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = true}, command_param_t{.u = 2},
                 command_param_t{.u = hidden_layer_size},
                 command_param_t{.u = hidden_layer_size}}});

  // bert_dense_meta_t v_meta(0, true, hidden_layer_size, hidden_layer_size);
  out.v_tid = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = true}, command_param_t{.u = 2},
                 command_param_t{.u = hidden_layer_size},
                 command_param_t{.u = hidden_layer_size}}});

  // bert_dense_meta_t multihead_out_meta(0, true, hidden_layer_size,
  // hidden_layer_size);
  out.multihead_out_tid = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = true}, command_param_t{.u = 2},
                 command_param_t{.u = hidden_layer_size},
                 command_param_t{.u = hidden_layer_size}}});

  // bert_dense_meta_t w1_meta(0, true, 4 * hidden_layer_size,
  // hidden_layer_size);
  out.w1_tid = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = true}, command_param_t{.u = 2},
                 command_param_t{.u = 4 * hidden_layer_size},
                 command_param_t{.u = hidden_layer_size}}});

  // bert_dense_meta_t w2_meta(0, true, hidden_layer_size, 4 *
  // hidden_layer_size);
  out.w2_tid = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = true}, command_param_t{.u = 2},
                 command_param_t{.u = hidden_layer_size},
                 command_param_t{.u = 4 * hidden_layer_size}}});

  return out;
}

tid_t generate_input_tensor(std::uint32_t batch_size, std::uint32_t seq_len,
                            std::uint32_t num_heads,
                            std::uint32_t hidden_layer_size,
                            std::vector<abstract_command_t> &commands) {

  // bert_dense_meta_t x_meta(0, false, batch_size, seq_len,
  // hidden_layer_size);
  tid_t out = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = false}, command_param_t{.u = 3},
                 command_param_t{.u = batch_size},
                 command_param_t{.u = seq_len},
                 command_param_t{.u = hidden_layer_size}}});
  return out;
}

tid_t generate_mask(std::uint32_t batch_size, std::uint32_t seq_len,
                    std::uint32_t num_heads, std::uint32_t hidden_layer_size,
                    std::vector<abstract_command_t> &commands) {

  // bert_dense_meta_t mask_meta(0, false, batch_size, seq_len, seq_len);
  tid_t out = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = GENERATE_TENSOR_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {},
      .output_tids = {current_tid++},
      .params = {command_param_t{.b = false}, command_param_t{.u = 3},
                 command_param_t{.u = batch_size},
                 command_param_t{.u = seq_len},
                 command_param_t{.u = seq_len}}});
  return out;
}

tid_t generate_transfomer(std::uint32_t batch_size, std::uint32_t seq_len,
                          std::uint32_t num_heads,
                          std::uint32_t hidden_layer_size,
                          transformer_weights_t weights, tid_t x, tid_t mask,
                          std::vector<abstract_command_t> &commands) {

  std::vector<bbts::command_param_t> _params = {
      bbts::command_param_t{.u = batch_size},
      bbts::command_param_t{.u = seq_len},
      bbts::command_param_t{.u = num_heads},
      bbts::command_param_t{.u = hidden_layer_size},
  };

  tid_t out = current_tid;
  commands.push_back(abstract_command_t{
      .ud_id = TRANSFORMER_ID,
      .type = abstract_command_type_t::APPLY,
      .input_tids = {x, mask, weights.q_tid, weights.k_tid, weights.v_tid,
                     weights.multihead_out_tid, weights.w1_tid, weights.w2_tid},
      .output_tids = {current_tid++},
      .params = _params});
  return out;
}

int main() {

  const std::uint32_t batch_size = 1;
  const std::uint32_t seq_len = 20;
  const std::uint32_t num_heads = 8;
  const std::uint32_t hidden_layer_size = 256;
  const std::uint32_t num_transformers = 4;

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  // specify functions
  funs.push_back(abstract_ud_spec_t{.id = GENERATE_TENSOR_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  funs.push_back(abstract_ud_spec_t{
      .id = TRANSFORMER_ID,
      .ud_name = "transformer",
      .input_types = {"bert_dense", "bert_dense", "bert_dense", "bert_dense",
                      "bert_dense", "bert_dense", "bert_dense", "bert_dense"},
      .output_types = {"bert_dense"}});

  tid_t x, mask;
  std::vector<transformer_weights_t> transformers;
  {
    std::vector<abstract_command_t> commands;
    x = generate_input_tensor(batch_size, seq_len, num_heads, hidden_layer_size,
                              commands);
    mask = generate_mask(batch_size, seq_len, num_heads, hidden_layer_size,
                         commands);

    for (auto idx = 0; idx < num_transformers; ++idx) {
      transformers.push_back(generate_transformer_tensors(
          batch_size, seq_len, num_heads, hidden_layer_size, commands));
    }

    // write out the commands
    std::ofstream gen("generate_tensors.sbbts");
    compile_source_file_t gsf{.function_specs = funs, .commands = commands};
    gsf.write_to_file(gen);
    gen.close();
  }

  {
    std::vector<abstract_command_t> commands;
    for (auto ws : transformers) {
      x = generate_transfomer(batch_size, seq_len, num_heads, hidden_layer_size,
                              ws, x, mask, commands);
    }

    // write out the commands
    std::ofstream gen("run.sbbts");
    compile_source_file_t gsf{.function_specs = funs, .commands = commands};
    gsf.write_to_file(gen);
    gen.close();
  }

  return 0;
}