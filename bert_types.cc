#include "bert_types.h"
#include <sstream>

namespace bert {

bbts::tensor_creation_fs_t bert_dense_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const bbts::tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(bert_dense_t *)here;
    auto &m = *(bert_dense_meta_t *)&_meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const bbts::tensor_meta_t &_meta) {
    auto &m = *(bert_dense_meta_t *)&_meta;

    auto num_elements = m.m().dim0 * m.m().dim1 * m.m().dim2;
    num_elements += m.m().has_bias ? m.m().dim0 : 0;

    // we add the bias if we have it to the size
    return sizeof(bbts::tensor_meta_t) + num_elements * sizeof(float);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    // get the tensor
    auto &t = *(bert_dense_t *)here;

    // extract the info
    auto dim0 = t.meta().m().dim0;
    auto dim1 = t.meta().m().dim1;
    auto dim2 = t.meta().m().dim2;
    auto data = t.data();

    // print the tensor
    auto idx = 0u;
    for (int i = 0; i < dim0; i++) {
      ss << "[";
      for (int j = 0; j < dim1; j++) {
        ss << "[";
        for (int j = 0; j < dim2; j++) {
          ss << data[idx++] << ((j == dim2 - 1) ? "" : ",");
        }
        ss << "]\n";
      }
      ss << "]\n";
    }
  };

  // return the tensor creation functions
  return bbts::tensor_creation_fs_t{
      .get_size = size, .init_tensor = init, .print = pnt};
}
} // namespace bert