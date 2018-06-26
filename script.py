import sim as s
import numpy as np
from scipy.integrate import trapz
from scipy.signal import medfilt, butter, bessel, lfilter, freqz, decimate, periodogram, welch
import matplotlib.pyplot as plt

# this function takes 1 ms window and produces spectrum.

# next, let's observe this as a periodogram using welch's method to get the averaged spectrum of multiple windows.
# t_flip/T_s = 1e-3*3e9 = 3 million samples each window. can we do this computation without creating an array each iteration...
# generator expression is like list comprehension with parentheses sum(a for i in list) 


jitter = 0.1e-12 # 10 picoseconds of RMS jitter

Fo=1.3e9
Fs = 3e9
Ts= 1/Fs

t_flip = 1e-3

binwidth = 1/t_flip

# number of samples (per channel) in the dataset. rounded to closest multiple of 4096 as allowed by the TSW14J56
# 32 GB of RAM. (can house up to 2,147,483,648 16 bit samples)


# scaling = 'density'

# if scaling == 'density' :
# 	binwidth = Fs/fft_len

# elif scaling == 'spectrum' :
# 	binwidth = 1

data = s.make_signal(t_j=jitter)
bins, power = periodogram(x=data, fs=Fs, window=None, nfft=len(data), return_onesided=True, scaling='density')


# from the frequency bins, let's integrate the spectrum to get the total amount of jitter... let's see how close this is to the
# value we input in the first place.

# this spectrum needs to be put into units of dBc/Hz, then converted to integrated dBc, 
# then we can convert to seconds or radians of jitter. Divide all of the bins by the maximum value.

# we'll do this outside of the function to check if the units are right in the ipython environment.
# this should be dBc/Hz. The question is if the order of operations matter for the bin scaling.
ssb_pn_lin = power/np.max(power)
maxpt = np.argmax(power)
print("maximum is at", maxpt*binwidth*1e-6, "MHz with V^2 value", power[maxpt])

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

# do calculation of total jitter contribution from the 

# 10 Hz to 10MHz
fo3 = 4e9
bins3=np.power(10, [1,2,3,4,5,6,7])
ssb_pn3_log = np.array([-65,-95,-102,-112,-117,-120,-132]) 
ssb_pn3_lin = np.power(10, (ssb_pn3_log/10))

area3 = 10*np.log10(trapz(y=ssb_pn3_lin, x=bins3))

tj3 = np.sqrt(2*10**(area3/10))/(2*np.pi*fo3)

# 10 Hz to 10MHz
fo4 = 1.8e9
bins4=np.array([12e3, 100e3, 1e6, 10e6, 20e6])
ssb_pn4_log = np.array([-118,-123,-141,-157,-157]) + 4.437
ssb_pn4_lin = np.power(10, (ssb_pn4_log/10))

area4 = 10*np.log10(trapz(y=ssb_pn4_lin, x=bins4))

tj4 = np.sqrt(2*10**(area4/10))/(2*np.pi*fo4)

#BNC device at 4GHz, 1kHz offset
fo5 = 4e9
bins5=np.array([1e3])
ssb_pn5_log = np.array([-108])
ssb_pn5_lin = np.power(10, (ssb_pn5_log/10))

area5 = 10*np.log10(bins5*ssb_pn5_lin)

tj5 = np.sqrt(2*10**(area5/10))/(2*np.pi*fo5)

# now let's convert the phase noise spectrum into an integrated value, then convert that into a jitter value in seconds.
# we want to integrate over a 1 kilohertz bandwidth to get this number, let's do just the two adjacent bins to the peak...
# or maybe several to be safe.

f_o = int(Fo/binwidth) 
integ = 10

area = 10*np.log10(trapz(y=ssb_pn_lin[f_o+12:f_o+int(20e3)], x=None, dx=binwidth))
jitter = np.sqrt(2*(10**(area/10)))/(2*np.pi*Fo)

print(jitter*1e12,"picoseconds of jitter recovered from integration")

fig, ax = plt.subplots(1,1)

ax.set_xlim(1e6, 1e10)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax.set_title('Phase Noise')
ax.step(bins, 10*np.log10(ssb_pn_lin))

fig2, ax2 = plt.subplots(1,1)

ax2.set_xscale('log')
#ax2.set_ylim()
ax2.set_xlabel('Frequency Offset (Hz)')
ax2.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax2.set_title('Phase Noise Spectrum')
ax2.step(bins[maxpt:]-Fo, 10*np.log10(ssb_pn_lin[maxpt:]))

# ax2.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


fig.show()
fig2.show()
input("press enter to finish")