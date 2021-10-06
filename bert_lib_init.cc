#include "bert_types.h"
#include "generate_tensor.h"
#include "third_party/bbts/src/tensor/tensor_factory.h"
#include "third_party/bbts/src/ud_functions/udf_manager.h"
#include "transformer.h"

extern "C" {

void register_tensors(bbts::tensor_factory_ptr_t tensor_factory) {
  tensor_factory->register_fmt("bert_dense",
                               bert::bert_dense_t::get_creation_fs());
}

void register_udfs(bbts::udf_manager_ptr udf_manager) {
  udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t{.ud_name = "transformer",
                      .is_ass = false,
                      .is_com = false,
                      .num_in = 8,
                      .num_out = 1,
                      .impls = {}}));
  udf_manager->register_udf_impl(std::make_unique<bert::transformer>());

  udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t{.ud_name = "generate_tensor",
                      .is_ass = false,
                      .is_com = false,
                      .num_in = 0,
                      .num_out = 1,
                      .impls = {}}));
  udf_manager->register_udf_impl(std::make_unique<bert::generate_tensor>());
}
}