"""compute symbolic state space model and jacobians

Notes
-----
The states without a thermal capacity must be placed on the left of
the incidence matrix !
"""
from sympy import *

init_printing(use_unicode=True)


# ----------------------------------------------------------------------------
# section to be modified by the user
Rw, Rb, Cw, Ci, Cb, Aw, Ai, cv, sc = symbols(
    'Rw, Rb, Cw, Ci, Cb, Aw, Ai, cv, sc')

R = diag(Rw / 2, Rw / 2, Rb / 2, Rb / 2)
C = diag(Cw * sc, Ci * sc, Cb * sc)
A = Matrix([[1, 0, 0],
            [-1, 1, 0],
            [0, -1, 1],
            [0, 0, 1]])
u = [True, False, False, True, True, True, False]
# ----------------------------------------------------------------------------

nb = R.shape[0]  # number of branches
nn = C.shape[0]  # number of nodes
nz = 0  # number of zeros
for i in range(nn):
    if C[i, i] == 0:
        nz += 1

G = R.inv()
Cc = C[nz:, nz:]

K = -A.T @ G @ A
K11 = K[:nz, :nz]
K12 = K[:nz, nz:]
K21 = K[nz:, :nz]
K22 = K[nz:, nz:]

Kb = A.T @ G
Kb1 = Kb[:nz, :]
Kb2 = Kb[nz:, :]

As = simplify(Cc.inv() @ (-K21 @ K11.inv() @ K12 + K22))
Bs0 = Cc.inv() @ (-K21 @ K11.inv() @ Kb1 + Kb2)
Bs1 = Cc.inv() @ -K21 @ K11.inv()
Bs2 = Cc.inv()
if Bs1.is_zero:
    Bs = Matrix(BlockMatrix([[Bs0, Bs2]]).as_explicit())
else:
    Bs = Matrix(BlockMatrix([[Bs0, Bs1, Bs2]]).as_explicit())

Bs = simplify(Bs[:, u])

# ----------------------------------------------------------------------------
# section to be modified by the user
Bs2 = zeros(3, 5)
Bs2[0, 0] = Bs[0, 0]
Bs2[0, 2] = Bs[0, 2] * Aw
Bs2[1, 2] = Bs[1, 3] * Ai
Bs2[1, 3] = Bs[1, 3]
Bs2[1, 4] = Bs[1, 3] * cv
Bs2[2, 1] = Bs[2, 1]
Bs = simplify(Bs2)

# Bs2 = Matrix(Bs.copy())
# Bs2[0, 1] = Bs[0, 1] * Aw
# Bs2[1, 1] = Bs[1, 2] * Ai
# Bs3 = zeros(2, 4)
# Bs3[:, :3] = Bs2
# Bs3[1, 3] = Bs3[1, 2] * cv
# # Bs2[:, :4] = Bs
# # Bs2[0, 2] *= Aw
# # Bs2[1, 2] = Bs2[1, 3] * Ai
# # Bs2[1, 4] = Bs2[1, 3] * cv
# Bs = simplify(Bs3)

print("\nstate matrix A")
print("-" * 15)
print(As)

print("\ninput matrix B")
print("-" * 15)
print(Bs)

for s in As.free_symbols:
    print(f"\ndA / d{s}")
    print("-" * 15)
    print(simplify(diff(As, s)))

for s in Bs.free_symbols:
    print(f"\ndB / d{s}")
    print("-" * 15)
    print(simplify(diff(Bs, s)))
