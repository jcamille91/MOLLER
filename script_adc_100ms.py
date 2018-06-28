import sim as s
import numpy as np
from scipy.integrate import trapz
from scipy.signal import medfilt, butter, bessel, lfilter, freqz, decimate, periodogram, welch
import matplotlib.pyplot as plt

def define_bessel_lpf(cutoff, fs, order=5) :
	nyq = 0.5 * fs
	normalized_cutoff = cutoff / nyq
	b, a = bessel(order, normalized_cutoff, btype='low', analog=False)
	return b, a
def A_resolution:(fo = 10e6, fs = 2949.12e6, bits = 12, int_time=1e-3, ncalc=100, file = '../ADC_DATA/something')
	
	''' 
	this function calculates the resolution of the ADC for measuring 
	the digitally down converted amplitudes (in 1ms windows)

	input:
	fo- linear frequency of the analog signal being sampled
	fs- sampling frequency of the analog to digital converter
	bits- number of bits precision used. adc has 12bit and 14bit modes.
	int_time- length of each integration window for calculating signal amplitudes in seconds.
	ncalc- number of amplitudes (integration windows) to calculate
	file- string of binary file containing data for channels A and B for the ADC
	'''	



	# ADC input voltage is the output code (16bit integer) * LSB
	# The Least Significant Bit is the full scale range divided by the number of available total bits (12 or 14 depending on mode)
	fsr = 1.35
	lsb_adc = fsr/(2**bits)
	maxcode = (2**bits)-1

		
	Wo = 2*np.pi*fo	
	Ts= 1/fs
	t_flip=1e-3

	spectrum = True
	ddc = False
	calc = False
	xcor = False

	binwidth = int(1/t_flip)

	l = int(t_flip*Fs)
	t = np.linspace(0, t_flip, int(t_flip/Ts))
	
	# binary channels are written [A,B]_0, [A,B]_1, [A,B]_2.... etc/
	# to choose a channel pick out every other linearly indexed data point
	data = s.read_binary(scale = lsb_adc, filename='../ADC_DATA/6_27_internal/6_27_2018_1d4GA_2949d12GS_littlesplitter.bin')
	#data = s.read_binary(scale = lsb_adc, filename='../ADC_DATA/6_27_external/')



	if spectrum :			
		bins_A, power_A = periodogram(x=data[0::2], fs=Fs, window=None, nfft=int(Fs*t_flip), return_onesided=True, scaling=scaling)
		bins_B, power_B = periodogram(x=data[1::2], fs=Fs, window=None, nfft=int(Fs*t_flip), return_onesided=True, scaling=scaling)

		pn_lin_A = (power_A/np.max(power_A))
		maxpt_A = np.argmax(power_A)
		pn_lin_B = (power_B/np.max(power_B))
		maxpt_B = np.argmax(power_B)

		print("Ch A tone is at", maxpt_A*binwidth*1e-6, "MHz")
		print("Ch B tone is at", maxpt_B*Fs/len(voltage_B)*1e-6, "MHz")

	
		fig_A, ax_A = plt.subplots(1,1)

		ax_A.set_xlim(1e6, 1e10)
		ax_A.set_xscale('log')
		#ax_A.set_yscale('log')
		ax_A.set_xlabel('Frequency (Hertz)')
		ax_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_A.set_title('CH A Noise Spectral Density')
		ax_A.step(bins_A, 10*np.log10(power_A/np.max(power_A))

		fig2_A, ax2_A = plt.subplots(1,1)

		ax2_A.set_xscale('log')
		ax2_A.set_xlim(1e3, 1e9)
		#ax2_A.set_ylim()
		ax2_A.set_xlabel('Frequency Offset (Hz)')
		ax2_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax2_A.set_title('CH A SSB Phase Noise')
		ax2_A.step(bins_A[maxpt_A:]-Fo, 10*np.log10(power_A/np.max(power_A)[maxpt_A:]))

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
		ax_B.step(bins_B, 10*np.log10(power_B/np.max(power_B)))

		fig2_B, ax2_B = plt.subplots(1,1)

		ax2_B.set_xscale('log')
		ax2_B.set_xlim(1e3, 1e9)
		# ax2_b.set_yscale('log')
		#ax2_B.set_ylim()
		ax2_B.set_xlabel('Frequency Offset (Hz)')
		ax2_B.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax2_B.set_title('CH B SSB Phase Noise ')
		ax2_B.step(bins_B[maxpt_B:]-Fo, 10*np.log10(power_B/np.max(power_B)[maxpt_B:])

		#ax2_B.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


		fig_B.show()
		fig2_B.show()

	if ddc :

		cutoff = 0.1e6
		b, a = define_bessel_lpf(cutoff=cutoff, fs=Fs, order=7)
		avg_A = np.zeros(ncalc)
		avg_B = np.zeros(ncalc)

		for i in range(ncalc) :

			start = i*l
			stop = (i+1)*l

			# get binary data as int16, scale by adc lsb. even data -> channel A , odd data -> channel B.
			# then multiply each 1ms window (of length l) by the sin and cos modulating terms.
			# t represents the appropriate time values. finally, use a lowpass filter on this modulated data.
			I_A = data[0::2][start:stop]*np.sin(Wo*t)
			Q_A = data[1::2][start:stop]*np.cos(Wo*t)

		# 	I_Af = lfilter(b, a, (lsb_adc*data[0::2][start:stop])*np.sin(Wo*t))
		# 	Q_Af = lfilter(b, a, (lsb_adc*data[0::2][start:stop])*np.cos(Wo*t))

		# 	# I_Bf = lfilter(b, a, lsb_adc*data[1::2][start:stop]*np.sin(Wo*t))
		# 	# Q_Bf = lfilter(b, a, lsb_adc*data[1::2][start:stop]*np.cos(Wo*t))

		# 	phase = np.arctan(-I_Af/Q_Af)	
		# 	avg_phase = np.mean(phase)
		# 	rms_phase = np.std(phase)

		# 	# average amplitude recovered from I,Q components
		# 	avg_A[i] = np.mean(2*(Q_Af*np.cos(avg_phase) - I_Af*np.sin(avg_phase)))
		# 	# a_B[i] = np.mean(2*(Q_Bf*np.cos(avg_phase) - I_Bf*np.sin(avg_phase)))

		# # sigma of the distribution of average amplitudes		
		# rms_A = np.std(avg_A)
		# rms_B = np.std(avg_B)

	if calc :
		pass

	if xcor :
		pass


	input("press enter to finish")

