import drjit as dr
import mitsuba as mi

from hashgrid import HashGrid

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
    n = 20
    sampler = mi.load_dict({"type": "independent"})  # type: mi.Sampler
    sampler.seed(0, n)
    sample = mi.Point3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())

    grid = HashGrid(sample, 3, 10)

    p = mi.Point3f(0.5, 0.5, 0.1)
    cell = grid.cell_idx(p)
    cell_size = dr.gather(mi.UInt, grid.cell_size, cell)
    offset = dr.gather(mi.UInt, grid.cell_offset, cell)
    idx = dr.gather(mi.UInt, grid.sample_idx, offset)
    x0 = dr.gather(mi.Point3f, sample, idx)
    print(f"{x0=}")
