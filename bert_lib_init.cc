#include "third_party/bbts/src/tensor/tensor_factory.h"
#include "third_party/bbts/src/ud_functions/udf_manager.h"
#include "bert_types.h"

extern "C" {

  void register_tensors(bbts::tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("bert_dense", bert::bert_dense_t::get_creation_fs());
  }
 
  void register_udfs(bbts::udf_manager_ptr udf_manager) {
  }
}