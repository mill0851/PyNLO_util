import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pynlo_util.pynlo_util as pyutil
import numpy as np
import pynlo

c = 299792458.0         # (m/s)
CENTER_WL = 3.1e-06     # (m)
GRID_SIZE = 1001        # unitless
FRAC_RANGE = 0.05       # unitless

omega_axis = pyutil.omega_grid(GRID_SIZE, CENTER_WL, FRAC_RANGE)    # (rad/s)
lambda_axis = pyutil.omega_to_wl(omega_axis)                        # (m)
n = pyutil.n_YAG(lambda_axis * 1e6)                                 # Unitless
beta0 = pyutil.propagation_const(omega_axis, n)                     # (1/m)
beta2, beta3, beta4, = pyutil.dispersion_coeff(beta0, omega_axis)   # (s^n/m)

print
print("Dispersion Coefficients for YAG at " + str(CENTER_WL) + " (m)")
print("Beta2 = " + str(beta2) + " (s^2/m) = " + str(beta2*1e27) + " (ps^2/km) or (fs^2/mm)")
print("Beta3 = " + str(beta3) + " (s^3/m) = " + str(beta3*1e39) + " (ps^3/km) or (fs^3/mm)")
print("Beta4 = " + str(beta4) + " (s^4/m) = " + str(beta4*1e51) + " (ps^4/km) or (fs^4/mm)")