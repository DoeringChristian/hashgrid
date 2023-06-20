import drjit as dr
import mitsuba as mi

from hashgrid import HashGrid
import hashgrid

# UInt = dr.cuda.UInt
#
# target = UInt(0, 0, 0, 0)
# idx = UInt(0, 0, 0, 0, 0, 0, 0)
#
# dst = hashgrid.scatter_atomic_inc(target, idx)
# print(f"{dst=}")
# print(f"{target=}")

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

    n = 1_000_000

    target = dr.zeros(mi.UInt, n)
    idx = dr.zeros(mi.UInt, n)

    dst = hashgrid.scatter_atomic_inc(target, idx)
    print(f"{target=}")
    print(f"{dst=}")
    print(f"{dr.sum(target)=}")

    # n = 1_000_000
    # for i in range(100):
    #     sampler = mi.load_dict({"type": "independent"})  # type: mi.Sampler
    #     sampler.seed(0, n)
    #     sample = mi.Point3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())
    #
    #     grid = HashGrid(sample, 3, 30)
    #     print(f"{grid.cell_size=}")
    #     print(f"{dr.sum(grid.cell_size)=}")
    #     print(f"{grid.cell_offset=}")

    # p = mi.Point3f(0.5, 0.5, 0.1)
    # cell = grid.cell_idx(p)
    # cell_size = dr.gather(mi.UInt, grid.cell_size, cell)
    # offset = dr.gather(mi.UInt, grid.cell_offset, cell)
    # idx = dr.gather(mi.UInt, grid.sample_idx, offset)
    # x0 = dr.gather(mi.Point3f, sample, idx)
    # print(f"{x0=}")
