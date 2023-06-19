import drjit as dr

import hashgrid

UInt = dr.cuda.UInt

target = UInt(0, 0, 0, 0)
idx = UInt(0, 0, 0, 0, 0, 0, 0)

dst = hashgrid.scatter_atomic_inc(target, idx)
print(f"{dst=}")
print(f"{target=}")
