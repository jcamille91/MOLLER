import sim as s
import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt

# this function takes 1 ms window and produces spectrum.

# next, let's observe this as a periodogram using welch's method to get the averaged spectrum of multiple windows.
# t_flip/T_s = 1e-3*3e9 = 3 million samples each window. can we do this computation without creating an array each iteration...
# generator expression is like list comprehension with parentheses sum(a for i in list) 


jitter = 10e-12 # 10 picoseconds of RMS jitter

Fo=1.3e9
Fs = 3e9
Ts= 1/Fs
window=1e-3
fft_len = int(window/Ts)

scaling = 'density'
bandwidth = [10**3, 10**9]

if scaling == 'density' :
	binwidth = Fs/fft_len

elif scaling == 'spectrum' :
	binwidth = 1

data = s.make_signal(t_j=jitter)
bins, power = s.spectrum(data, Fs = Fs, fft_len=window, scaling = scaling)

# from the frequency bins, let's integrate the spectrum to get the total amount of jitter... let's see how close this is to the
# value we input in the first place.

# this spectrum needs to be put into units of dBc/Hz, then converted to integrated dBc, 
# then we can convert to seconds or radians of jitter. Divide all of the bins by the maximum value.

# we'll do this outside of the function to check if the units are right in the ipython environment.
# this should be dBc/Hz. The question is if the order of operations matter for the bin scaling.
ssb_pn_lin = power/np.max(power)
maxpt = np.argmax(power)
print("maximum is at", maxpt*binwidth*1e-9, "GHz with V^2 value", power[maxpt])

# replicate MT-008 Analog Devices guide data to verify that integration is working correctly.
# single sided spectrum

#-150db is 1e-15
Fs2 = 100e6
Ts2=1/Fs2
fft2len = 1e6 # frequency binwidth is 100Hz 
bins2 = np.arange(10e3,200e6, 100)
ssb_pn2 = np.ones(len(bins2))*1e-15

area2 = 10*np.log10(trapz(y=ssb_pn2, x=None, dx=100))
tj2 = np.sqrt(2*10**(area2/10))/(2*np.pi*100e6)


# now let's convert the phase noise spectrum into an integrated value, then convert that into a jitter value in seconds.
# we want to integrate over a 1 kilohertz bandwidth to get this number, let's do just the two adjacent bins to the peak...
# or maybe several to be safe.

integ_pt_off = 10
integ_pt = np.arange(maxpt, maxpt+integ_pt_off)
area = (trapz(y=ssb_pn_lin[integ_pt], x=None, dx=binwidth))
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

ax2.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


fig.show()
fig2.show()
input("press enter to finish")