import sim as s
import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt

# this function takes binary codes from the ADC32RF45 and produces a spectrum.

# ADC input voltage is the output code (16bit integer) * LSB
# The Least Significant Nit is the full scale range divided by the number of available total bits
lsb_adc = 1.35/(2**12)


Channel = 'A'
Fo=10e6
Fs = 2949.12e6 # ADC32RF45 sampling frequency (from LMX2582 synth)
Ts= 1/Fs

window=1e-3
fft_len = int(window/Ts)

scaling = 'density'
bandwidth = [10**3, 10**9]

# binary channels are written [A,B]_0, [A,B]_1, [A,B]_2.... etc/
# to choose a channel one, pick out every other linearly indexed data point
# if you don't you'll plot both.

if Channel = 'A' :
	choose = 0
elif Channel = 'B' :
	choose = 1



if scaling == 'density' :
	binwidth = Fs/fft_len

elif scaling == 'spectrum' :
	binwidth = 1

data = s.read_binary('../ADC_DATA/10Minput_2949.12Msample_bypass_12bit.bin')

codes = data[choose::2]
voltages = codes*lsb_adc

bins, power = s.spectrum(data, Fs = Fs, fft_len=len(data), scaling = scaling)

# from the frequency bins, let's integrate the spectrum to get the total amount of jitter... let's see how close this is to the
# value we input in the first place.

# this spectrum needs to be put into units of dBc/Hz, then converted to integrated dBc, 
# then we can convert to seconds or radians of jitter. Divide all of the bins by the maximum value.

# we'll do this outside of the function to check if the units are right in the ipython environment.
# this should be dBc/Hz. The question is if the order of operations matter for the bin scaling.
ssb_pn = (10*np.log10(power/np.max(power)))
maxpt = np.argmax(power)
print("maximum is at", maxpt*binwidth*1e-9, "GHz with V^2 value", power[maxpt])

# replicate MT-008 Analog Devices guide data to verify that integration is working correctly.
#ssb_pn2 = np.ones()

# now let's convert the phase noise spectrum into an integrated value, then convert that into a jitter value in seconds.
# we want to integrate over a 1 kilohertz bandwidth to get this number, let's do just the two adjacent bins to the peak...
# or maybe several to be safe.

integ_pt_off = 10
integ_pt = np.arange(maxpt, maxpt+integ_pt_off)
area = (trapz(y=ssb_pn[integ_pt], x=None, dx=1))
#area += 10*np.log10(bandwidth[1]-bandwidth[0])
jitter = np.sqrt(2*(10**(area/10)))/(2*np.pi*Fo)

print(jitter*1e12,"picoseconds of jitter recovered from integration")

fig, ax = plt.subplots(1,1)

ax.set_xlim(1e6, 1e10)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax.set_title('Phase Noise')
ax.step(bins, ssb_pn)

fig2, ax2 = plt.subplots(1,1)

ax2.set_xscale('log')
#ax2.set_ylim()
ax2.set_xlabel('Frequency Offset (Hz)')
ax2.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax2.set_title('Phase Noise Spectrum')
ax2.step(bins[maxpt:]-Fo, ssb_pn[maxpt:])

#ax2.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


fig.show()
fig2.show()
input("press enter to finish")