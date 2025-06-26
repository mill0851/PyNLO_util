import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pynlo_util.pynlo_util as pyutil
import numpy as np
import pynlo
import matplotlib.pyplot as plt

# Here I calculate the effective nonlinearity as a function of distance from the focus
# based on what the beams cross sectional area is at these points

c = 299792458.0         # (m/s)
CENTER_WL = 3.1e-06     # (m)
BEAM_WAIST = 50e-06     # (m) beam radius at focus
MAX_Z = 0.005           # (m)
Z_STEP = 0.0001         # (m)
RAYLEIGH_LENGTH = (np.pi * BEAM_WAIST**2) / (CENTER_WL) # (m)
KERR_COEFF = 6e-20      # (m^2/W)

z_axis, beam_radius = pyutil.map_beam_radius(BEAM_WAIST, RAYLEIGH_LENGTH, MAX_Z, Z_STEP, verbose=False)
beam_area = np.pi * beam_radius**2
gamma = (2 * np.pi * KERR_COEFF) / (CENTER_WL * beam_area)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)

ax1.plot(z_axis*1e3, beam_radius*1e6, label="Beam Radius", color='red')
ax1.set_ylabel("Beam Radius (um)", fontsize=16)
ax1.set_xlabel("Distance from Focus (mm)", fontsize=16)
ax1.set_title("Beam Radius vs Distance from Focus", fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=14)

ax2.plot(z_axis*1e3, gamma, label="Effective Nonlinearity", color='red')
ax2.set_ylabel("Effective Nonlinearity (1/(m)(W))", fontsize=16)
ax2.set_xlabel("Distance from Focus (mm)", fontsize=16)
ax2.set_title("Effective Nonlinearity vs Distance from Focus", fontsize=16)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='both', labelsize=14)

plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

# for the YAG experiment we need the gamma values at 0-3mm with a 0.5mm step
indices = [0]
distances = np.arange(0.0005, 0.0035, 0.0005)
for i in distances: 
    N = i/Z_STEP
    indices.append(int(N))

gamma_experiment = []
for i in indices:
    gamma_experiment.append(round(gamma[i],8))

print(gamma_experiment)
