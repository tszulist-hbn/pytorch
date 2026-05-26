/*
 * CPU implementation of _to_sparse_semi_structured
 * This implementation performs the sparse semi-structured conversion on CPU.
 */

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <tuple>

namespace at::native {

std::tuple<Tensor, Tensor>
_to_sparse_semi_structured(const Tensor& dense) {
  // Check dimensions of the dense matrix.
  TORCH_CHECK(dense.dim() == 2,
              __func__, " : Expected dense argument to be 2D tensor, got ",
              dense.dim(), " dims");

  // Determine PyTorch datatype for the metadata matrix.
  auto meta_dtype = at::kChar;
  auto ksparse = 0;
  auto dense_elems_per_meta_elem = 0;
  if (dense.dtype() == at::kChar) {
    meta_dtype = at::kInt;
    ksparse = 4;
    dense_elems_per_meta_elem = 32;
  } else if (dense.dtype() == at::kHalf || dense.dtype() == at::kBFloat16) {
    meta_dtype = at::kShort;
    ksparse = 4;
    dense_elems_per_meta_elem = 16;
  } else if (dense.dtype() == at::kFloat) {
    meta_dtype = at::kShort;
    ksparse = 2;
    dense_elems_per_meta_elem = 8;
  } else {
    TORCH_CHECK(false, "_to_sparse_semi_structured: Invalid dense argument datatype ",
             dense.dtype(), " encountered");
  }

  const auto dense_nrows = dense.size(0);
  const auto dense_ncols = dense.size(1);

  if (dense_nrows % (meta_dtype == at::kShort ? 32 : 16) != 0) {
    TORCH_CHECK(false, "_to_sparse_semi_structured: Number of rows of dense matrix must "
             "be divisible by ", (meta_dtype == at::kShort ? 32 : 16),
             ", but it is ", dense_nrows);
  }
  if (dense_ncols % dense_elems_per_meta_elem != 0) {
    TORCH_CHECK(false, "_to_sparse_semi_structured: Number of columns of dense matrix "
             "must be divisible by ", dense_elems_per_meta_elem, ", but it is ",
             dense_ncols);
  }

  const auto dense_cpu = dense.to("cpu");

  const auto mask_cpu = dense_cpu != at::zeros({1}, dense_cpu.options());

  const auto sparse_cpu =
    dense_cpu.masked_select(mask_cpu).view({dense_nrows, dense_ncols / 2});

  const auto meta_nrows = dense_nrows;
  const auto meta_ncols = dense_ncols / dense_elems_per_meta_elem;
  auto meta_cpu = dense_cpu.new_empty({meta_nrows, meta_ncols},
                                      at::TensorOptions().dtype(meta_dtype));

  auto* mask_cpu_ptr = mask_cpu.data_ptr<bool>();
  for (auto i = 0; i < meta_nrows; ++i) {
    for (auto j = 0; j < meta_ncols; ++j) {
      uint64_t meta_val = 0;
      for (auto k = 0; k < dense_elems_per_meta_elem / ksparse; ++k, mask_cpu_ptr += ksparse) {
        const auto mask_elems =
          (ksparse == 4) ? std::make_tuple(mask_cpu_ptr[0], mask_cpu_ptr[1],
                                           mask_cpu_ptr[2], mask_cpu_ptr[3])
                         : std::make_tuple(mask_cpu_ptr[0], mask_cpu_ptr[0],
                                           mask_cpu_ptr[1], mask_cpu_ptr[1]);
        auto meta_quadruple = 0;
        if (mask_elems == std::make_tuple(1, 1, 0, 0)) {
          meta_quadruple = 4; // 0100
        } else if (mask_elems == std::make_tuple(1, 0, 1, 0)) {
          meta_quadruple = 8; // 1000
        } else if (mask_elems == std::make_tuple(0, 1, 1, 0)) {
          meta_quadruple = 9; // 1001
        } else if (mask_elems == std::make_tuple(1, 0, 0, 1)) {
          meta_quadruple = 12; // 1100
        } else if (mask_elems == std::make_tuple(0, 1, 0, 1)) {
          meta_quadruple = 13; // 1101
        } else if (mask_elems == std::make_tuple(0, 0, 1, 1)) {
          meta_quadruple = 14; // 1110
        } else {
          TORCH_CHECK(false, "_to_sparse_semi_structured: dense argument does not match ",
                   (dense.dtype() != at::kFloat) ? "2:4" : "1:2",
                   "sparsity pattern");
        }
        meta_val = meta_val | (meta_quadruple << (4 * k));
      }
      const auto idx = i * meta_ncols + j;
      if (meta_dtype == at::kShort) {
        using MetaElement = int16_t;
        const auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
        meta_cpu_ptr[idx] = (MetaElement)meta_val;
      } else if (meta_dtype == at::kInt) {
        using MetaElement = int32_t;
        const auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
        meta_cpu_ptr[idx] = (MetaElement)meta_val;
      }
    }
  }

  auto meta_reordered_cpu = meta_cpu.new_empty({meta_nrows, meta_ncols});
  // Simple row-major to column-major interleaved reordering
  if (meta_dtype == at::kShort) {
    using MetaElement = int16_t;
    auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
    auto meta_reordered_cpu_ptr = meta_reordered_cpu.data_ptr<MetaElement>();
    for (auto i = 0; i < meta_nrows; ++i) {
      for (auto j = 0; j < meta_ncols; ++j) {
        const auto src_idx = i * meta_ncols + j;
        const auto dst_idx = (j * 2 + (i % 2)) * (meta_nrows / 2) + (i / 2);
        meta_reordered_cpu_ptr[dst_idx] = meta_cpu_ptr[src_idx];
      }
    }
  } else if (meta_dtype == at::kInt) {
    using MetaElement = int32_t;
    auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
    auto meta_reordered_cpu_ptr = meta_reordered_cpu.data_ptr<MetaElement>();
    for (auto i = 0; i < meta_nrows; ++i) {
      for (auto j = 0; j < meta_ncols; ++j) {
        const auto src_idx = i * meta_ncols + j;
        const auto dst_idx = (j * 2 + (i % 2)) * (meta_nrows / 2) + (i / 2);
        meta_reordered_cpu_ptr[dst_idx] = meta_cpu_ptr[src_idx];
      }
    }
  }

  return std::make_tuple(sparse_cpu.to(dense.device()),
                         meta_reordered_cpu.to(dense.device()));
}

}  // namespace at::native
