from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import drjit as dr

from ._hashgrid_core import scatter_atomic_inc_uint_cuda


def scatter_atomic_inc(target: dr.cuda.UInt, idx: dr.cuda.UInt):
    UInt = type(target)
    assert dr.is_cuda_v(target) and dr.is_cuda_v(idx)
    assert dr.is_unsigned_v(target) and dr.is_unsigned_v(idx)
    assert dr.is_integral_v(target) and dr.is_integral_v(idx)

    n_values = dr.shape(idx)[-1]
    dst = dr.empty(type(idx), n_values)  # type: dr.cuda.UInt
    dr.scatter(dst, 0, dr.arange(UInt, n_values))

    if idx.data_() == target.data_():
        idx_ = dr.empty(UInt, n_values)
        dr.scatter(
            idx_,
            idx,
            dr.arange(UInt, n_values),
        )
        idx = idx_

    dr.eval(target, idx, dst)
    dr.sync_thread()

    assert (
        dst.data_() != idx.data_()
        and dst.data_() != target.data_()
        and target.data_() != idx.data_()
    )

    scatter_atomic_inc_uint_cuda(target.data_(), idx.data_(), dst.data_(), n_values)

    # SAFETY: access values to keep variables alive
    assert target.data_() > 0 and dst.data_() > 0 and idx.data_() > 0

    return dst
