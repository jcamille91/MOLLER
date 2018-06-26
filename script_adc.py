import sim as s
import numpy as np
from scipy.integrate import trapz
from scipy.signal import periodogram
import matplotlib.pyplot as plt

# this function takes binary codes from the ADC32RF45 and produces a spectrum.

# ADC input voltage is the output code (16bit integer) * LSB
# The Least Significant Bit is the full scale range divided by the number of available total bits (12 or 14 depending on mode)
lsb_adc = 1.35/(2**12)
fsr = 1.35
maxcode = (2**12)-1

Fo=10e6		
Fs = 2949.12e6 # ADC32RF45 sampling frequency (from LMX2582 synth)
Ts= 1/Fs


scaling = 'density'



# binary channels are written [A,B]_0, [A,B]_1, [A,B]_2.... etc/
# to choose a channel pick out every other linearly indexed data point


# data = s.read_binary('../ADC_DATA/10Minput_2949.12Msample_bypass_12bit_CHAB_NOSYNC_100ms.bin')
data = s.read_binary('../ADC_DATA/10Minput_2949.12Msample_bypass_12bit_CHAB_NOSYNC_1ms.bin')
# data = s.read_binary('../ADC_DATA/10Minput_2949.12Msample_bypass_12bit_CHAB.bin')

voltage_A = lsb_adc*data[0::2]
voltage_B = lsb_adc*data[1::2]


if scaling == 'density' : # V^2 / Hz
	binwidth = 1

elif scaling == 'spectrum' : # V^2
	binwidth = Fs/len(voltage_A)



# bins_A, power_A = s.spectrum(voltage_A, Fs = Fs, npt=len(voltage_A), scaling = scaling)
# bins_B, power_B = s.spectrum(voltage_B, Fs = Fs, npt=len(voltage_B), scaling = scaling)

bins_A, power_A = periodogram(x=voltage_A, fs=Fs, window=None, nfft=len(voltage_A), return_onesided=True, scaling=scaling)
bins_B, power_B = periodogram(x=voltage_B, fs=Fs, window=None, nfft=len(voltage_B), return_onesided=True, scaling=scaling)

# scale by binwidth before normalizing to the carrier
power_A = power_A/binwidth
power_B = power_A/binwidth

pn_lin_A = (power_A/np.max(power_A))
maxpt_A = np.argmax(power_A)
pn_lin_B = (power_B/np.max(power_B))
maxpt_B = np.argmax(power_B)

print("Ch A tone is at", maxpt_A*Fs/len(voltage_A)*1e-6, "MHz")
print("Ch B tone is at", maxpt_B*Fs/len(voltage_B)*1e-6, "MHz")

# calculate the close in phase noise and broadband phase noise for each channel

# 10 MHz offset (20MHz absolute) is about location where spectrum is flattened (broadband phase noise)
f_bb = int(20e6*len(voltage_A)/Fs)+1
f_o = int(Fo*len(voltage_A)/Fs)+1
area_bb = 10*np.log10(trapz(y=pn_lin_A[f_bb:], x=None, dx=Fs/len(voltage_A)))
area_ci = 10*np.log10(trapz(y=pn_lin_A[f_o:f_bb], x=None, dx=Fs/len(voltage_A)))
jitter_bb = np.sqrt(2*(10**(area_bb/10)))/(2*np.pi*Fo)
jitter_ci = np.sqrt(2*(10**(area_ci/10)))/(2*np.pi*Fo)

# closein = 10
# broadband = 10
# integ_pt = np.arange(maxpt, maxpt+integ_pt_off)
# area_ci = 10*np.log10(trapz(y=pn_lin_A[integ_pt], x=None, dx=binwidth))
# area_bb = 10*np.log10(trapz(y=pn_lin_A[integ_pt], x=None, dx=binwidth))
# jitter = np.sqrt(2*(10**(area/10)))/(2*np.pi*Fo)

# print(jitter*1e12,"picoseconds of jitter recovered from integration")

fig_A, ax_A = plt.subplots(1,1)

ax_A.set_xlim(1e6, 1e10)
ax_A.set_xscale('log')
#ax_A.set_yscale('log')
ax_A.set_xlabel('Frequency (Hertz)')
ax_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax_A.set_title('CH A Noise Spectral Density')
ax_A.step(bins_A, 10*np.log10(pn_lin_A))

fig2_A, ax2_A = plt.subplots(1,1)

ax2_A.set_xscale('log')
ax2_A.set_xlim(1e3, 1e9)
#ax2_A.set_ylim()
ax2_A.set_xlabel('Frequency Offset (Hz)')
ax2_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax2_A.set_title('CH A SSB Phase Noise')
ax2_A.step(bins_A[maxpt_A:]-Fo, 10*np.log10(pn_lin_A[maxpt_A:]))

#ax2_A.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


fig_A.show()
fig2_A.show()

fig_B, ax_B = plt.subplots(1,1)

ax_B.set_xlim(1e6, 1e10)
ax_B.set_xscale('log')
#ax_B.set_yscale('log')
ax_B.set_xlabel('Frequency (Hertz)')
ax_B.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax_B.set_title('CH B Noise Spectral Density')
ax_B.step(bins_B, 10*np.log10(pn_lin_B))

fig2_B, ax2_B = plt.subplots(1,1)

#ax2_B.set_xscale('log')
ax2_B.set_xlim(1e3, 1e9)
# ax2_b.set_yscale('log')
#ax2_B.set_ylim()
ax2_B.set_xlabel('Frequency Offset (Hz)')
ax2_B.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
ax2_B.set_title('CH B SSB Phase Noise ')
ax2_B.step(bins_B[maxpt_B:]-Fo, 10*np.log10(pn_lin_B[maxpt_B:]))

#ax2_B.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


fig_B.show()
fig2_B.show()

input("press enter to finish")