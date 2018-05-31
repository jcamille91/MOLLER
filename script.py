import sim as s
import numpy as np
import matplotlib.pyplot as plt

# this function takes 1 ms window and produces spectrum.

# next, let's observe this as a periodogram using welch's method to get the averaged spectrum of multiple windows.
# t_flip/T_s = 1e-3*3e9 = 3 million samples each window. can we do this computation without creating an array each iteration...
# generator expression is like list comprehension with parentheses sum(a for i in list) 

Fs = 3e9
Ts= 1/Fs
window=1e-3
fft_len = int(window/Ts)
binwidth = Fs/fft_len


data = s.make_signal(t_j=10e-12) # 1 picosecond of RMS jitter
bins, power = s.spectrum(data, Fs = Fs, fft_len=window, scaling = 'spectrum')

# from the frequency bins, let's integrate the spectrum to get the total amount of jitter... let's see how close this is to the
# value we input in the first place.

# this spectrum needs to be put into units of dBc/Hz, then converted to integrated dBc, 
# then we can convert to seconds or radians of jitter. Divide all of the bins by the maximum value.

# we'll do this outside of the function to check if the units are right in the ipython environment.
# this should be dBc/Hz. The question is if the order of operations matter for the bin scaling.
ssb_pn = (20*np.log10(power/np.max(power)))/binwidth
maxpt = np.argmax(power)
print("maximum is at", maxpt*binwidth*1e-9, "GHz with V^2 value", power[maxpt])

fig, ax = plt.subplots(1,1)

ax.set_xlim(1e6, 1e10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel(r'$\frac{V^2}{Hz}$', rotation=0, fontsize=16)
ax.set_title('Power Spectral Density')
ax.step(bins, power)

fig2, ax2 = plt.subplots(1,1)

ax2.set_xscale('log')
#ax2.set_ylim()
ax2.set_xlabel('Frequency (Hertz)')
ax2.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax2.set_title('Power Spectral Density')
ax2.step(bins, ssb_pn)


fig.show()
fig2.show()
input("press enter to finish")