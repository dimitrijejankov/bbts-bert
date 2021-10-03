#pragma once

#include "third_party/bbts/src/tensor/tensor.h"

namespace bert {

struct bert_dense_meta_t : public bbts::tensor_meta_t {

  // the meta stucture
  struct m_t {

    uint32_t num_dim;

    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;

    bool has_bias;
  };

  // returns the meta data struct
  m_t &m() const {

    // we use it as the blob
    return *((m_t *)_blob);
  }

  // init the tensor with the format impl_id
  bert_dense_meta_t(bbts::tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  bert_dense_meta_t(bbts::tfid_t _id, bool has_bias, uint32_t dim0)
      : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_dim = 1, .dim0 = dim0, .dim1 = 1, .dim2 = 1};
  }

  bert_dense_meta_t(bbts::tfid_t _id, bool has_bias, uint32_t dim0,
                    uint32_t dim1)
      : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_dim = 2, .dim0 = dim0, .dim1 = dim1, .dim2 = 1};
  }

  // init the tensor meta with row and column numbers
  bert_dense_meta_t(bbts::tfid_t _id, bool has_bias, uint32_t dim0,
                    uint32_t dim1, uint32_t dim2)
      : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_dim = 3, .dim0 = dim0, .dim1 = dim1, .dim2 = dim2};
  }
};

struct bert_dense_t : public bbts::tensor_t {

  // return the meta data of the dense tensor
  bert_dense_meta_t &meta() const { return *((bert_dense_meta_t *)&_meta); }

  // return the
  float *data() const { return (float *)_blob; }

  // returns the bias
  float *bias() const {
    return ((float *)_blob) +
           meta().m().dim0 * meta().m().dim1 * meta().m().dim2;
  }

  // return creation functions
  static bbts::tensor_creation_fs_t get_creation_fs();
};

} // namespace bert