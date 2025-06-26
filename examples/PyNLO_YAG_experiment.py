import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pynlo_util.pynlo_util as pyutil
import numpy as np
import matplotlib.pyplot as plt
import pynlo
from scipy.signal import hilbert
from matplotlib.colors import LogNorm

# Input Pulse Parameters
########################
FWHM_DURATION = 0.07    # (ps)
CENTER_WL = 3100        # (nm)
PULSE_ENERGY = 3e-06    # (J)
GDD = 0.0               # (ps^2)
TOD = 0.0               # (ps^3)
# PEAK_POWER = 0.94 * PULSE_ENERGY / (FWHM_DURATION * 1e-12)  # (J/s)

# Grid Parameters
#################
TIME_SPAN = 10.0       # (ps)
PROP_STEPS = 50     # This is how many steps the Z distance is divided into
TIME_STEPS = 2**13  # This is how many temporal points there are

# Fiber Parameters
##################
FIBER_LENGTH = 0.003    # (m)
KERR_COEFF = 6e-20      # (m^2/W)
ALPHA = 0.0             # attenuation const (dB/cm)
BETA2 = -408.15         # (ps^2/km)
BETA3 = 2.54            # (ps^3/km)
BETA4 = -0.02           # (ps^4/km)
GAMMA = [1.548 * 1e-05, 1.490 * 1e-05, 1.340 * 1e-05, 1.147 * 1e-05, 0.954 * 1e-05, 0.785 * 1e-05, 0.645 * 1e-05]   # (1/Wm)
RAMAN = True
STEEP = True


# Create Pulse
##############
input_pulse = pynlo.light.DerivedPulses.GaussianPulse(1, FWHM_DURATION, CENTER_WL, time_window_ps = TIME_SPAN,
                                                GDD = GDD, TOD = TOD, NPTS = TIME_STEPS, frep_MHz = 100, power_is_avg = False)
input_pulse.set_epp(PULSE_ENERGY)


# Create Fiber and Loop Through Gamma Values
############################################
output_pulses = []
results = []

for idx, gamma_val in enumerate(GAMMA):
    print("Simulating gamma value " + str(idx+1) + " of " + str(len(GAMMA)) + " gamma = " + str(gamma_val))
    
    # Create fiber
    fiber = pynlo.media.fibers.fiber.FiberInstance()
    fiber.generate_fiber(FIBER_LENGTH, center_wl_nm=CENTER_WL, betas=(BETA2, BETA3, BETA4),
                        gamma_W_m=gamma_val, gvd_units="ps^n/km", gain=-ALPHA)

    # Initiate propagation
    evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.001, USE_SIMPLE_RAMAN=True,
                                                     disable_Raman=np.logical_not(RAMAN),
                                                     disable_self_steepening=np.logical_not(STEEP))
    y, AW, AT, pulse_out = evol.propagate(pulse_in=input_pulse, fiber=fiber, n_steps=PROP_STEPS)
    
    # Store Results
    output_pulses.append(pulse_out)
    results.append({
        'gamma': gamma_val,
        'y': y * 1e3,  # Convert distance to mm
        'AW': AW,
        'AT': AT,
        'pulse_out': pulse_out
    })


# PLOTS
#######
# Plot spectrum evolution and final spectral content for each simulation
for idx, res in enumerate(results):
    # Extract data for current gamma
    gamma_val = res['gamma']
    y = res['y']
    AW = res['AW']
    AT = res['AT']
    pulse_out = res['pulse_out']
    
    # Get frequency and convert to wavelength
    F = input_pulse.W_mks / (2 * np.pi) * 1e-12  # Frequency in THz
    
    # Process data
    zW = pyutil.dB(np.transpose(AW)[:, (F > 0)])
    zT = pyutil.dB(np.transpose(AT))

    # Create Figure
    fig = plt.figure(figsize=(10,10))
    ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
    ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
    ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)
    
    ax0.plot(F[F > 0], zW[-1]/np.max(zW[-1]), color='r')
    ax1.plot(input_pulse.T_ps, zT[-1], color='r')

    ax0.plot(F[F > 0],   zW[0]/np.max(zW[0]), color='b')
    ax1.plot(input_pulse.T_ps, zT[0], color='b')

    extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, FIBER_LENGTH * 1e3)
    ax2.imshow(zW, extent=extent, vmin=np.max(zW) - 60.0,
                 vmax=np.max(zW), aspect='auto', origin='lower')

    extent = (np.min(input_pulse.T_ps), np.max(input_pulse.T_ps), np.min(y), FIBER_LENGTH * 1e3)
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


# Plot the power curve for each pulse including the input pulse
input_power = np.abs(input_pulse.AT)**2
time_ps = input_pulse.T_ps
input_pulse_duration = pyutil.fwhm(time_ps, input_power)
print("input pulse duration: " + str(input_pulse_duration) + " (ps)")

plt.figure(figsize=(10,5))
plt.plot(time_ps, input_power, color='black', label='Input Pulse')

pulse_durations = []
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange']
labels = ['0 mm', '0.5 mm', '1.0 mm', '1.5 mm', '2.0 mm', '2.5 mm', '3.0 mm']
for idx, pulse in enumerate(output_pulses):
    label = labels[idx]
    output_power = np.abs(pulse.AT)**2
    pulse_duration = pyutil.fwhm(time_ps, output_power)
    pulse_durations.append(pulse_duration)
    plt.plot(time_ps, output_power, color=colors[idx], label=label)

plt.xlabel('Time (ps)', fontsize=14)
plt.ylabel('Instantaneous Power (W)', fontsize=14)
plt.title('Instantaneous Power vs Time', fontsize=16)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# Plot the pulse duration as a function of distance from the focus
distances = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
pulse_durations_hemmer_converging = [0.0335, 0.0325, 0.0335, 0.0375, 0.052, 0.065, 0.066] # estimation from figures in hemmer et al.
pulse_durations_hemmer_diverging = [0.0335, 0.035, 0.04125, 0.053, 0.0595, 0.064, 0.067]
plt.figure(figsize=(10,5))
plt.plot(distances, pulse_durations, color='black', label='PyNLO')
plt.plot(distances, pulse_durations_hemmer_converging, color='blue', label='Hemmer, converging')
plt.plot(distances, pulse_durations_hemmer_diverging, color='red', label='Hemmer, diverging')
plt.xlabel("Distance from Focus (mm)", fontsize=14)
plt.ylabel("Pulse Duration (ps)", fontsize=14)
plt.title("Pulse Duration vs Distance from Focus", fontsize=16)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()