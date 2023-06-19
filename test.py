import drjit as dr

import hashgrid

UInt = dr.cuda.UInt

target = dr.detach(UInt(0, 0, 0))
value = dr.detach(UInt(1, 1, 1))
idx = dr.detach(UInt(0, 0, 1))

dr.eval(target, value, idx)

dst = hashgrid.scatter_atomic(target, value, idx)
