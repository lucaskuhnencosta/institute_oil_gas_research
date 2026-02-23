import casadi as ca
from Formal_Surrogate_ODE_Model.glc_01_casadi import glc_casadi
import numpy as np

y0 = ca.DM([3919.7688, 437.16663, 7956.1206])
u0 = ca.DM([0.2, 0.8])

y = ca.MX.sym("y",3); u = ca.MX.sym("u",2)
dx,z = glc_casadi(y,u)
F = ca.Function("F",[y,u],[dx,z])

dx0,z0 = F(y0,u0)
print("dx0:", np.array(dx0).squeeze())
print("z0 :", np.array(z0).squeeze())
