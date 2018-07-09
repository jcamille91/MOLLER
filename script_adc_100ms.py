import numpy as np
from scipy.integrate import trapz
from scipy.signal import medfilt, butter, bessel, lfilter, freqz, decimate, periodogram, welch
import matplotlib.pyplot as plt

def define_bessel_lpf(cutoff, fs, order=5, btype='low') :
	nyq = 0.5 * fs
	normalized_cutoff = cutoff / nyq
	b, a = bessel(order, normalized_cutoff, btype='low', analog=False)
	return b, a

def plot(d, x, axis = None) :
	''' simple plot function. supply an axis object to add to an already existing plot.
	*** Recommended to plot less than a million points or matplotlib blows up sometimes. ***
	
	input :
	d : n-length 1-dimensional numpy array
	x :  x-axis
	npt_max : max number of array points to plot to avoid matplotlib crash.
	axis : matplotlib axis object for plotting to.
	'''

	npt = len(x)

	if isinstance(axis, ax_obj) :	# if an axis is supplied, plot to it.
		axis.step(x, d[:npt])

	else :	# no axis, make a quick standalone plot.
		plt.step(x, d[:npt])
		plt.show()

		input('press enter to close plot')
		
		plt.close()

def read_binary(scale, filename = '../data/test') :
	'''
	this reads a binary file interpreted as series of 16bit integers, as is the case for our ADC's binary codes
	'''
	data = scale*np.fromfile(filename, dtype=np.int16, count=-1, sep="")
	return data

def A_resolution(fo = 10e6, fs = 2850e6, bits = 12, int_time=1e-3, ncalc=100, 
	spectrum=False, ddc=False, ddc2=False, calc=False, xcor=False, plot=False,
	file = '../ADC_DATA/7_5_external/7_5_2018_10MA_2850MS_extclk.bin') :
	
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

# fo = 10e6
# fs = 2850e6
# bits = 12
# int_time=1e-3 
# ncalc=100
# spectrum=False 
# ddc=True
# calc=False
# xcor=False 
# plot=True
# file = '../ADC_DATA/6_27_internal/6_27_2018_1d4GA_2949d12GS_littlesplitter.bin'
# file = '../ADC_DATA/6_27_external/6_27_2018_10MA_2850GS_clkfilter.bin'
# file = '../ADC_DATA/7_2_internal/7_2_2018_1d4GA_2949d12GS_littlesplitter_5dBm_run2.bin'
# file = '../ADC_DATA/7_5_internal/7_5_2018_1024MA_3072MS_15dBm.bin'
# file = '../ADC_DATA/7_5_external/7_5_2018_10MA_2850MS_extclk.bin'
# ADC input voltage is the output code (16bit integer) * LSB
# The Least Significant Bit is the full scale range divided by the number of available total bits (12 or 14 depending on mode)
	fsr = 1.35
	lsb_adc = fsr/(2**bits)
	maxcode = (2**bits)-1
	nch = 2 # ADC32RF45 has 2 separate channels
		
	Wo = 2*np.pi*fo	
	Ts= 1/fs


	binwidth = int(1/int_time)

	# binary channels are written [A,B]_0, [A,B]_1, [A,B]_2.... etc/
	# to choose a channel pick out every other linearly indexed data point
	data = read_binary(scale = lsb_adc, filename=file)



	if spectrum :	

		bins_A, power_A = periodogram(x=data[0::2], fs=fs, window=None, nfft=int(fs*int_time), return_onesided=True, scaling='density')
		bins_B, power_B = periodogram(x=data[1::2], fs=fs, window=None, nfft=int(fs*int_time), return_onesided=True, scaling='density')

		pn_lin_A = (power_A/np.max(power_A))
		maxpt_A = np.argmax(power_A)
		pn_lin_B = (power_B/np.max(power_B))
		maxpt_B = np.argmax(power_B)

		print("Ch A tone is at", maxpt_A*binwidth*1e-6, "MHz")
		print("Ch B tone is at", maxpt_B*binwidth*1e-6, "MHz")


		fig_A, ax_A = plt.subplots(1,1)

		ax_A.set_xlim(1e6, 1e10)
		ax_A.set_xscale('log')
		#ax_A.set_yscale('log')
		ax_A.set_xlabel('Frequency (Hertz)')
		ax_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_A.set_title('CH A Noise Spectral Density')
		ax_A.step(bins_A, 10*np.log10(power_A/np.max(power_A)))

		fig2_A, ax2_A = plt.subplots(1,1)

		ax2_A.set_xscale('log')
		ax2_A.set_xlim(1e3, 1e9)
		#ax2_A.set_ylim()
		ax2_A.set_xlabel('Frequency Offset (Hz)')
		ax2_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax2_A.set_title('CH A SSB Phase Noise')
		ax2_A.step(bins_A[maxpt_A:]-fo, 10*np.log10((power_A/np.max(power_A))[maxpt_A:]))

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
		ax2_B.step(bins_B[maxpt_B:]-fo, 10*np.log10((power_B/np.max(power_B))[maxpt_B:]))

		#ax2_B.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


		fig_B.show()
		fig2_B.show()

	if ddc2 :

		# this filter is to remove spurs after 10MHz for the reference input.
		cutoff0 = 13e6

		b0, a0 = define_bessel_lpf(cutoff=cutoff0, fs=fs, order=3)
		
		# this lowpass filter is for the digital downconversion
		cutoff = 1e4

		b1, a1 = define_bessel_lpf(cutoff=cutoff, fs=fs, order=3)


		# arrays to store average amplitudes (2 channels)
		avg = np.zeros((nch,ncalc))
		rms = np.zeros(nch)
		# number of samples in 1ms window
		l = int(int_time*fs)
		t_phi_avg = ncalc*l # averaging time for the phase. set to integer multiple of the integration window.

		#rad = Wo*np.linspace(0, int_time, l)
		rad = Wo/fs*np.arange(l) # this way gets the phase values more accurately than linspace 2piFo/Fs*n

		# look up table for sin and cosine values for 1ms window
		I_phi=np.sin(rad) 
		Q_phi=np.cos(rad)

		if fo < 12e6 :
			# lowpass filtered data for each channel
			Ch = np.array([lfilter(b0, a0, data[0::2][:l]), lfilter(b0, a0, data[1::2][:l])])

		else :
			Ch = np.array([data[0::2], data[1::2]])

		I_v=np.zeros(len(Ch[0]))
		Q_v=np.zeros(len(Ch[0]))

		for k in range(nch):
	
			for i in range(ncalc) :

				start = i*l
				stop = (i+1)*l

				# get binary data as int16, scale by adc lsb. even data -> channel A , odd data -> channel B.
				# then multiply each 1ms window (of length l) by the sin and cos modulating terms.
				# t represents the appropriate time values. finally, use a lowpass filter on this modulated data.
				I_v[start:stop] = Ch[k][start:stop]*I_phi
				Q_v[start:stop] = Ch[k][start:stop]*Q_phi

				I_vf = lfilter(b1, a1, I_v)
				Q_vf = lfilter(b1, a1, Q_v)
		

			phase = np.arctan(-I_Af[:t_phi_avg]/Q_Af[:t_phi_avg])	
			avg_phase = np.mean(phase)

			for i in range(ncalc) :
				start = i*l
				stop = (i+1)*l
				# average amplitude recovered from I,Q components
				avg[k][i] = np.mean(2*(Q_vf[start:stop]*np.cos(avg_phase) - I_vf[start:stop]*np.sin(avg_phase)))

			# sigma of the distribution of average amplitudes		
			rms[k] = np.std(avg[k])
			print(rms[k], 'sigma error')


	if ddc :

		cutoff0 = 13e6

		b0, a0 = define_bessel_lpf(cutoff=cutoff0, fs=fs, order=3)
		
		cutoff = 1e4
		b, a = define_bessel_lpf(cutoff=cutoff, fs=fs, order=3)
		w, h = freqz(b, a, worN=3000000)
		figf, axf = plt.subplots(1,1)
		axf.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
		axf.plot(cutoff, 0.5*np.sqrt(2), 'ko')
		axf.axvline(cutoff, color='k')
		axf.set_xlim(0, 0.5*fs)
		axf.set_title("Lowpass Filter Frequency Response")
		axf.set_xlabel('Frequency [Hz]')
		axf.grid()
		figf.show()

		dcavgpt = int(3e7)
		# arrays to store average amplitudes
		avg_A = np.zeros(ncalc)
		avg_B = np.zeros(ncalc)

		l = int(int_time*fs)
		t_phi_avg = 100*int(int_time*fs) # averaging time for the phase. set to integer multiple of the integration window.

		#rad = Wo*np.linspace(0, int_time, l)
		rad = Wo/fs*np.arange(l) # this way gets the phase values more accurately than linspace 2piFo/Fs*n
		ChA = lfilter(b0, a0, data[0::2]) # - np.mean(data[0::2][:dcavgpt])
		# ChA = data[0::2]

		# ChB=data[1::2] # - np.mean(data[1::2][:dcavgpt])
		I_A = np.zeros(len(ChA))
		Q_A = np.zeros(len(ChA))

		I=np.sin(rad)
		Q=np.cos(rad)

		for i in range(ncalc) :

			start = i*l
			stop = (i+1)*l

			# get binary data as int16, scale by adc lsb. even data -> channel A , odd data -> channel B.
			# then multiply each 1ms window (of length l) by the sin and cos modulating terms.
			# t represents the appropriate time values. finally, use a lowpass filter on this modulated data.
			I_A[start:stop] = ChA[start:stop]*I
			Q_A[start:stop] = ChA[start:stop]*Q
			# print(i)

		I_Af = lfilter(b, a, I_A)
		# print('done i filter')
		Q_Af = lfilter(b, a, Q_A)
		# print('done q filter')
		# # # 	# I_Bf = lfilter(b, a, lsb_adc*data[1::2][start:stop]*np.sin(Wo*t))
		# # # 	# Q_Bf = lfilter(b, a, lsb_adc*data[1::2][start:stop]*np.cos(Wo*t))

		phase = np.arctan(-I_Af[:t_phi_avg]/Q_Af[:t_phi_avg])	
		avg_phase = np.mean(phase)
		# # rms_phase = np.std(phase)

		for i in range(ncalc) :
			start = i*l
			stop = (i+1)*l
		# average amplitude recovered from I,Q components
			avg_A[i] = np.mean(2*(Q_Af[start:stop]*np.cos(avg_phase) - I_Af[start:stop]*np.sin(avg_phase)))
			# print(i, 'calc #')
		# # 	# a_B[i] = np.mean(2*(Q_Bf*np.cos(avg_phase) - I_Bf*np.sin(avg_phase)))

		# # # # sigma of the distribution of average amplitudes		
		rms_A = np.std(avg_A)
		print(rms_A, 'sigma error')
		# # # rms_B = np.std(avg_B)

		if plot :

			pltlen = 900000
			xaxis = np.arange(pltlen)*Ts*1.0e6

			fig1, ax1 = plt.subplots(1,1)	
			ax1.set_xlabel('microseconds')
			ax1.set_ylabel('amplitude')
			ax1.set_title('I component')
			
			fig2, ax2 = plt.subplots(1,1)
			ax2.set_xlabel('microseconds')
			ax2.set_ylabel('amplitude')
			ax2.set_title('Q component')
			
			fig3, ax3 = plt.subplots(1,1)
			ax3.set_xlabel('microseconds')
			ax3.set_ylabel('amplitude')
			ax3.set_title('I component (filtered)')
			
			fig4, ax4 = plt.subplots(1,1)
			ax4.set_xlabel('microseconds')
			ax4.set_ylabel('amplitude')
			ax4.set_title('Q component (filtered)')
			
			# fig5, ax5 = plt.subplots(1,1)
			# ax5.set_xlabel('microseconds')
			# ax5.set_ylabel('amplitude')
			# ax5.set_title('Reconstructed Amplitude')

			
			s.plot(I_A[:len(xaxis)], x = xaxis, axis=ax1)
			s.plot(Q_A[:len(xaxis)], x = xaxis, axis=ax2)
			s.plot(I_Af[:len(xaxis)], x = xaxis, axis=ax3)
			s.plot(Q_Af[:len(xaxis)], x = xaxis, axis=ax4)
			# s.plot(a, x = xaxis, axis=ax5)

			

			fig1.show()
			fig2.show()
			fig3.show()
			fig4.show()
			# fig5.show()

	if calc :
		pass

	if xcor :
		h, xbins, ybins = np.historgram2d(x=avg_A, y=avg_B, bins=None, range=None, normed=None)
		figc, axc = plt.subplots(1,1)

	input("press enter to finish")
	plt.close('all')
	return avg
