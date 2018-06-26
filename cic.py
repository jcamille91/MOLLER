# Math
import numpy as np
from scipy.signal import medfilt, butter, bessel, lfilter, freqz, decimate, periodogram, welch
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter

# Plotting
import matplotlib.pyplot as plt
from matplotlib import axes
ax_obj = axes.Axes
	
fs = 3e9

# prototype a cascaded comb integrator filter (linear phase low pass FIR filter)
#cutoff = 0.1e9
R = 8 # rate change, decimation ratio (reduces fs/R -> fs')
M = 1 # differential delay (filter order)

b = np.array([1,-1])	# numerator
a = np.array([1])			# denominator

# plot filter frequency response to check results are as expected

w, h = freqz(b, a, worN=3000000)
angles = np.unwrap(np.angle(h))

figf, axf = plt.subplots(1,1)
axf.plot(0.5*fs*w/np.pi, np.abs(h), 'r')
# axf.plot(cutoff, 0.5*np.sqrt(2), 'ko')
# axf.axvline(cutoff, color='k')
axf.set_xlim(0, 0.5*fs)
axf.set_title("CIC Filter Magnitude Response")
axf.set_xlabel('Frequency [Hz]')

axf.grid()
figf.show()

figp, axp = plt.subplots(1,1)
axp.plot(0.5*fs*w/np.pi, angles, 'g')
# axp.plot(cutoff, 0.5*np.sqrt(2), 'ko')
# axp.axvline(cutoff, color='k')
axp.set_xlim(0, 0.5*fs)
axp.set_title("CIC Filter Phase Response")
axp.set_xlabel('Frequency [Hz]')

axp.grid()
figp.show()

input("press enter to finish")
plt.close('all')