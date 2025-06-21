import pynlo_util.pynlo_util as pyutil
import numpy as np
import matplotlib.pyplot as plt
import pynlo

c = 299792458.0 # m/s

# Pulse Parameters
FWHM_DURATION = 0.07    # (ps)
CENTER_WL = 3100        # (nm)
PULSE_ENERGY = 3e-06    # (J)
GDD = 0.0               # (ps^2)
TOD = 0.0               # (ps^3)
PEAK_POWER = 0.94 * PULSE_ENERGY / (FWHM_DURATION * 1e-12)  # (J/s)
AREA = np.pi * (5e-06)**2   # Effective area for fiber (beam at focus here) (m^2)

# Grid Parameters
TIME_SPAN = 2        # (ps)
PROP_STEPS = 50
TIME_STEPS = 2**13

# Fiber Parameters
FIBER_LENGTH = 3.0      # (mm)
KERR_COEFF = 6e-20
ALPHA = 0.0             # attenuation const (dB/cm)
GAMMA = 5 * ((KERR_COEFF * pyutil.wl_to_omega(CENTER_WL * 1e-9)) / (c * AREA)) * 1e3 # Effective Nonlinearity (1/(W km))
print(GAMMA)
print()
RAMAN = True
STEEP = True

# Calculate Dispersion Coefficients
w = pyutil.omega_grid(1001, CENTER_WL * 1e-9, 0.05)
wl = pyutil.omega_to_wl(w)
n = pyutil.n_YAG(wl * 1e6)
beta0 = pyutil.propagation_const(w, n)
beta2, beta3, beta4 = pyutil.dispersion_coeff(beta0, w)


print("Beta2 = " + str(beta2 * 1e27))
print("Beta3 = " + str(beta3 * 1e39))
print("Beta4 = " + str(beta4 * 1e51))
print()

beta2 = beta2 * 1e27
beta3 = beta3 * 1e39
beta4 = beta4 * 1e51

# set up plots for the results:
fig = plt.figure(figsize=(10,10))
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)

# CREATE PULSE
pulse = pynlo.light.DerivedPulses.GaussianPulse(PEAK_POWER, FWHM_DURATION/1.76, CENTER_WL, time_window_ps = TIME_SPAN,
                                                GDD = GDD, TOD = TOD, NPTS = TIME_STEPS, frep_MHz = 100, power_is_avg = False)
pulse.set_epp(PULSE_ENERGY)

# CREATE FIBER
fiber = pynlo.media.fibers.fiber.FiberInstance()
fiber.generate_fiber(FIBER_LENGTH * 1e-3, center_wl_nm = CENTER_WL, betas = (beta2, beta3, beta4), gamma_W_m = GAMMA * 1e-3,
                     gvd_units = "ps^n/km", gain = -ALPHA)

# PROPAGATE
evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error = 0.001, USE_SIMPLE_RAMAN = True,
                                                   disable_Raman = np.logical_not(RAMAN),
                                                   disable_self_steepening = np.logical_not(STEEP))
y, AW, AT, pulse_out = evol.propagate(pulse_in = pulse, fiber = fiber, n_steps = PROP_STEPS)


F = pulse.W_mks / (2 * np.pi) * 1e-12 # convert to THz

def dB(num):
    return 10 * np.log10(np.abs(num)**2)

zW = dB( np.transpose(AW)[:, (F > 0)] )
zT = dB( np.transpose(AT) )

y = y * 1e3 # convert distance to mm


ax0.plot(F[F > 0],  zW[-1], color='r')
ax1.plot(pulse.T_ps,zT[-1], color='r')

ax0.plot(F[F > 0],   zW[0], color='b')
ax1.plot(pulse.T_ps, zT[0], color='b')


extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, FIBER_LENGTH)
ax2.imshow(zW, extent=extent, vmin=np.max(zW) - 60.0,
                 vmax=np.max(zW), aspect='auto', origin='lower')

extent = (np.min(pulse.T_ps), np.max(pulse.T_ps), np.min(y), FIBER_LENGTH)
ax3.imshow(zT, extent=extent, vmin=np.max(zT) - 60.0,
           vmax=np.max(zT), aspect='auto', origin='lower')


ax0.set_ylabel('Intensity (dB)')

ax2.set_xlabel('Frequency (THz)')
ax3.set_xlabel('Time (ps)')

ax2.set_ylabel('Propagation distance (mm)')

ax2.set_xlim(0,400)

ax0.set_ylim(-80,0)
ax1.set_ylim(-40,40)

plt.show()