import pickle
import numpy as np
from scipy.fftpack import fft
from scipy.integrate import trapz
from scipy.signal import medfilt, butter, bessel, lfilter, freqz, decimate, periodogram, welch
import matplotlib.pyplot as plt
from matplotlib import axes
ax_obj = axes.Axes

def define_bessel_lpf(cutoff, fs, order=5, btype='low') :
	nyq = 0.5 * fs
	normalized_cutoff = cutoff / nyq
	b, a = bessel(order, normalized_cutoff, btype='low', analog=False)
	return b, a

def calculate_jitter(ssb_pn, fbins, carrier, units='log') :

	'''function to calculate rms jitter (time domain expression of phase noise).
	input: 

	ssb_pn_log- single side band phase noise, relative to the carrier.
	expressed in decibels relative to the carrier in a 1 Hz bandwidth. [dBc/Hz]
	*** dBc = 10*log10(P/Pc). binwidth scaling needs to happen before the logarithm and carrier power normalization.
	fbins- linear frequency bins associated with each phase noise value provided
	carrier- linear frequency value associated with the carrier being referenced for the ssb phase noise values.
	units- choose lienar or logarithmic units of phase noise
	output:
	tj_rms- rms value of the jitter, integrated over bandwidth of the phase noise
	'''

	if units == 'log' : # use the logarithmic values
		ssb_pn_log = np.array(ssb_pn)
		fbins = np.array(fbins)
		ssb_pn_lin = np.power(10, ssb_pn_log/10)
		integrated_pn = 10*np.log10(trapz(y=ssb_pn_lin, x=fbins))
		tj_rms = np.sqrt(2*10**(integrated_pn/10))/(2*np.pi*carrier)

	elif units == 'lin' :	# use the linear values
		ssb_pn_lin= np.array(ssb_pn)
		fbins=np.array(fbins)
		integrated_pn = 10*np.log10(trapz(y=ssb_pn_lin, x=fbins))
		tj_rms = np.sqrt(2*10**(integrated_pn/10))/(2*np.pi*carrier)

	return tj_rms

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

def read_binary(infile, outfile, bits=12, fsr=1.35, raw=None) :
	'''
	this reads a binary file interpreted as series of 16bit integers, as is the case for our ADC's binary codes.
	two arrays of data are returned, for the adc's channel A and channel B.
	input:
	filename- binary file containing channels A and B. The data is organized:
	(CHA[n=0], CHB[n=0]), (CHA[n=1], CHB[n=1]), ..., (CHA[n=N], CHB[n=N])
	bits- number of bits used by the ADC to construct the data
	fsr- full scale range of the ADC in volts peak to peak
	raw- Set to true to return the raw ADC codes instead of converted voltage values
	'''

	if raw :
		out = np.fromfile(infile, dtype=np.int16, count=-1, sep="")

	else :
		data = (np.fromfile(infile, dtype=np.int16, count=-1, sep="")-(2**(bits-1)))*fsr/(2**bits)
		out = np.stack((data[0::2], data[1::2]))
		np.save(outfile, out)

def open_binary(infile) :
	data = np.load(infile)
	return data[0], data[1]

def make_fshift(outfile, fo, fs, int_time) :

	l = int(fs*int_time)
	Wo = 2*np.pi*fo
	rad = Wo/fs*np.arange(l)
	out = np.stack((np.sin(rad), np.cos(rad)))
	np.save(outfile, out)


def read_fshift(infile) :
	data = np.load(infile)
	sin = data[0]
	cos = data[1]
	return sin, cos

def spectrum2(file, fo = 10e6, fs = 2850e6, nbits=12, int_time=1e-3, n_window=99, plot=False) :
	'''calculate the fourier spectrum of the adc's digitized signal.
	convert spectrum to units of dBc/Hz (decibels normalized to the carrier power in 1Hz bandwidth).
	calculate the jitter contribution in some specified bandwidth relative to the carrier.
	input:
	file- string giving binary file containing data
	fo- expected carrier frequency
	fs- ADC sampling clock frequency
	int_time- integration time in seconds.
	int_bw- bandwidth for jitter calculation specified in integer multiples of the binwidth. binwidth = 1/(int_time) Hz
	This bandwidth is single sideband, relative to the carrier (specified in frequency offset, not absolute frequency).
	'''

	# the channel codes are converted to volts, unless 'raw=True'
	ChA, ChB = read_binary(filename=file, nbits=nbits, fsr=1.35, raw=None)
	N = int(fs*int_time)	# number of samples in integration window.

	binwidth = int(1/int_time) # Hz
	# this indexes the bins to get the desired frequency bins for integrating the phase noise
	# index = np.linspace(int(int_bw[0]), int(int_bw[1]), int(int_bw[1]-int_bw[0])+1, dtype=int)

	Saa = np.zeros(int(N/2)) # power spectrum of channel A
	Sbb = np.zeros(int(N/2)) # power spectrum of channel B
	Sba = np.zeros(int(N/2), dtype=np.complex128) # cross correlation spectrum of channels A and B

	for i in range(n_window):
		print(i)
		start = int(i*N)
		stop = int((i+1)*N)

		# get positive frequencies of FFT, normalize by N
		a = fft(ChA[start:stop])[:int(N/2)]/N
		b = fft(ChB[start:stop])[:int(N/2)]/N
		
		# sum the uncorrelated variances
		Saa += np.square(np.abs(a))
		Sbb += np.square(np.abs(b))
		Sba += b*np.conj(a)

	# divide by the binwidth and the number of spectrums averaged. multiply by 2 for single sided spectrum.
	# This single-sided power spectral density has units of volts^2/Hz
	Saa = 2*Saa/n_window/binwidth
	Sbb = 2*Sbb/n_window/binwidth

	# each cross correlation spectrum needs complex numbers to be averaged
	# because each calculation uses a complex conjugate.
	# wait to convert to a real PSD until the averaging is complete.
	# this spectrum is due to correlated noise sources.
	Sba = 2*np.abs(Sba)/n_window/binwidth

	fbins = np.linspace(0, fs/2, int(N/2))

	# This spectrum is due to only uncorrelated noise sources.
	Sdiff = (Saa + Sbb - 2*np.abs(Sba))

	# cutoff0 = 13e6
	# if fo < cutoff0 :
	# 	b0, a0 = define_bessel_lpf(cutoff=cutoff0, fs=fs, order=3)
	# 	ChA = lfilter(b0, a0, ChA)
	# 	ChB = lfilter(b0, a0, ChB)
	if plot:

		fig_A, ax_A = plt.subplots(1,1)

		ax_A.set_xlim(1e6, 1e10)
		ax_A.set_xscale('log')
		#ax_A.set_yscale('log')
		ax_A.set_xlabel('Frequency (Hertz)')
		ax_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_A.set_title('CH A Noise Spectral Density')
		ax_A.step(fbins, 10*np.log10(Saa/np.max(Saa)))

		fig_B, ax_B = plt.subplots(1,1)

		ax_B.set_xlim(1e6, 1e10)
		ax_B.set_xscale('log')
		#ax_B.set_yscale('log')
		ax_B.set_xlabel('Frequency (Hertz)')
		ax_B.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_B.set_title('CH B Noise Spectral Density')
		ax_B.step(fbins, 10*np.log10(Sbb/np.max(Sbb)))

		fig_C, ax_C = plt.subplots(1,1)

		ax_C.set_xlim(1e6, 1e10)
		ax_C.set_xscale('log')
		#ax_C.set_yscale('log')
		ax_C.set_xlabel('Frequency (Hertz)')
		ax_C.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_C.set_title('A/B Cross Correlation Noise Spectral Density')
		ax_C.step(fbins, 10*np.log10(np.abs(Sba)/np.max(Sba)))

		fig_D, ax_D = plt.subplots(1,1)

		ax_D.set_xlim(1e6, 1e10)
		ax_D.set_xscale('log')
		#ax_D.set_yscale('log')
		ax_D.set_xlabel('Frequency (Hertz)')
		ax_D.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_D.set_title('Phase Noise Spectral Density')
		ax_D.step(fbins, 10*np.log10(Sdiff/np.max(Sdiff)))

		fig_A.show()
		fig_B.show()
		fig_C.show()
		fig_D.show()

	tone_a = np.argmax(Saa)*binwidth
	tone_b = np.argmax(Sbb)*binwidth

	print('Channel A:', tone_a, 'Hz')
	print('Channel B:', tone_b, 'Hz')

	# tj_A = calculate_jitter(ssb_pn=pn_lin_A[index], fbins=bins_A[index], carrier=fo, units='lin')
	# tj_B = calculate_jitter(ssb_pn=pn_lin_B[index], fbins=bins_B[index], carrier=fo, units='lin')


def spectrum(file, fo = 10e6, fs = 2850e6, int_time=1e-3, int_bw=[1,10e3], plot=False) :
	'''calculate the fourier spectrum of the adc's digitized signal.
	convert spectrum to units of dBc/Hz (decibels normalized to the carrier power in 1Hz bandwidth).
	calculate the jitter contribution in some specified bandwidth relative to the carrier.
	input:
	file- string giving binary file containing data
	fo- expected carrier frequency
	fs- ADC sampling clock frequency
	int_time- integration time in seconds.
	int_bw- bandwidth for jitter calculation specified in integer multiples of the binwidth. binwidth = 1/(int_time) Hz
	This bandwidth is single sideband, relative to the carrier (specified in frequency offset, not absolute frequency).
	'''
	
	ChA, ChB = read_binary(filename=file)
	n_sample_window = int(fs*int_time)	# number of samples in 1ms integration window.

	binwidth = int(1/int_time) # Hz
	index = np.linspace(int(int_bw[0]), int(int_bw[1]), int(int_bw[1]-int_bw[0])+1, dtype=int) # this indexes the bins to get the desired frequency bins
				  																			   # for integrating the phase noise
	cutoff0 = 13e6
	if fo < cutoff0 :
		b0, a0 = define_bessel_lpf(cutoff=cutoff0, fs=fs, order=3)
		ChA = lfilter(b0, a0, ChA)
		ChB = lfilter(b0, a0, ChB)

	# throw out the first window of data, the filter needs response needs to stabilize.
	bins_A, power_A = periodogram(x=ChA, fs=fs, nfft=n_sample_window, return_onesided=True, scaling='density')
	bins_B, power_B = periodogram(x=ChB, fs=fs, nfft=n_sample_window, return_onesided=True, scaling='density')

	pn_lin_A = (power_A/np.max(power_A))
	maxpt_A = np.argmax(power_A)
	pn_lin_B = (power_B/np.max(power_B))
	maxpt_B = np.argmax(power_B)

	# tj_A = calculate_jitter(ssb_pn=pn_lin_A[index], fbins=bins_A[index], carrier=fo, units='lin')
	# tj_B = calculate_jitter(ssb_pn=pn_lin_B[index], fbins=bins_B[index], carrier=fo, units='lin')

	print("Ch A tone is at", maxpt_A*binwidth*1e-6, "MHz")
	# print("Ch A has", tj_A*1e15, "femtoseconds of jitter")
	print("Ch B tone is at", maxpt_B*binwidth*1e-6, "MHz")
	# print("Ch B has", tj_B*1e15, "femtoseconds of jitter")

	if plot :
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
		ax2_A.step(bins_A[maxpt_A:]-fo, 10*np.log10((pn_lin_A)[maxpt_A:]))

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

		ax2_B.set_xscale('log')
		ax2_B.set_xlim(1e3, 1e9)
		# ax2_b.set_yscale('log')
		#ax2_B.set_ylim()
		ax2_B.set_xlabel('Frequency Offset (Hz)')
		ax2_B.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax2_B.set_title('CH B SSB Phase Noise ')
		ax2_B.step(bins_B[maxpt_B:]-fo, 10*np.log10((pn_lin_B)[maxpt_B:]))

		#ax2_B.scatter(integ_pt*1e3, ssb_pn[integ_pt], marker='x', c='r')


		fig_B.show()
		fig2_B.show()
	# return bins_A, index

def ddc(ChA, ChB, sin, cos, fo = 10e6, fs = 3000e6, bits = 12, int_time=1e-3, ncalc=100, nch=2, 
	ddc=False, xcor=False, plot=False) :
	
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
		
	Wo = 2*np.pi*fo	
	Ts= 1/fs


	binwidth = int(1/int_time)

	# binary channels are written [A,B]_0, [A,B]_1, [A,B]_2.... etc/
	# to choose a channel pick out every other linearly indexed data point
	# if not (ChA or ChB) :
	# 	ChA, ChB = read_binary(filename=file, nbits=bits, fsr=1.35, raw=None)

	if ddc :

		# this filter is to remove spurs after 10MHz for the reference input.
		cutoff0 = 13e6

		b0, a0 = define_bessel_lpf(cutoff=cutoff0, fs=fs, order=3)
		
		# this lowpass filter is for the digital downconversion
		cutoff = 1e4

		b1, a1 = define_bessel_lpf(cutoff=cutoff, fs=fs, order=3)
		# w, h = freqz(b1, a1, worN=3000000)
		# figf, axf = plt.subplots(1,1)
		# axf.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
		# axf.plot(cutoff, 0.5*np.sqrt(2), 'ko')
		# axf.axvline(cutoff, color='k')
		# axf.set_xlim(0, 0.5*fs)
		# axf.set_title("Lowpass Filter Frequency Response")
		# axf.set_xlabel('Frequency [Hz]')
		# axf.grid()
		# figf.show()

		# arrays to store average amplitudes and sigma (2 channels)
		avg = np.zeros((nch,ncalc))
		rms = np.zeros(nch)

		# number of samples in 1ms window
		l = int(int_time*fs)
		npt = ncalc*l # averaging time for the phase. set to integer multiple of the integration window.

		#rad = Wo*np.linspace(0, int_time, l)

		# if not sin or cos :
		# 	rad = Wo/fs*np.arange(l) # this way gets the phase values more accurately than linspace 2piFo/Fs*n

		# 	# look up table for sin and cosine values for 1ms window
		# 	I_phi=np.sin(rad) 
		# 	Q_phi=np.cos(rad)

		# if fo < 12e6 :
		# 	# lowpass filtered data for each channel, the 10MHz data is very noisy.
		# 	Ch = np.array([lfilter(b0, a0, ChA[:npt]), lfilter(b0, a0, ChB[:npt])])
		# 	# for now, now more filtering for the high frequency data. maybe bandpass if necessary.
		# else :
		Ch = np.array([ChA[:npt], ChB[:npt]])

		I=np.zeros(len(Ch[0]))
		Q=np.zeros(len(Ch[0]))

		for k in range(nch):
	
			for i in range(ncalc) :
				# we do i+1 onward to avoid the first dataset being affected by the lowpass filter settling time.
				# start = i+1*l
				# stop = (i+2)*l

				start = i*l
				stop = (i+1)*l

				# get binary data as int16, scale by adc lsb. even data -> channel A , odd data -> channel B.
				# then multiply each 1ms window (of length l) by the sin and cos modulating terms.
				# t represents the appropriate time values. finally, use a lowpass filter on this modulated data.
				I[start:stop] = Ch[k][start:stop]*sin
				Q[start:stop] = Ch[k][start:stop]*cos

			I_f = lfilter(b1, a1, I)
			Q_f = lfilter(b1, a1, Q)
		

			phase = np.arctan(-I_f[:npt]/Q_f[:npt])	
			avg_phase = np.mean(phase)

			for i in range(ncalc) :
				start = i*l
				stop = (i+1)*l
				# average amplitude recovered from I,Q components
				avg[k][i] = np.mean(2*(Q_f[start:stop]*np.cos(avg_phase) + I_f[start:stop]*np.sin(avg_phase)))

			# sigma of the distribution of average amplitudes		
			rms[k] = np.std(avg[k])
			print('sigma error =', rms[k])


	
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

			
			plot(I[:len(xaxis)], x = xaxis, axis=ax1)
			plot(Q[:len(xaxis)], x = xaxis, axis=ax2)
			plot(I_f[:len(xaxis)], x = xaxis, axis=ax3)
			plot(Q_f[:len(xaxis)], x = xaxis, axis=ax4)
			# s.plot(a, x = xaxis, axis=ax5)

			

			fig1.show()
			fig2.show()
			fig3.show()
			fig4.show()
			# fig5.show()

	if xcor :
		r = [[min(avg[0]), max(avg[0])], [min(avg[1]), max(avg[1])]]
		h, xbins, ybins = np.histogram2d(x=avg_A, y=avg_B, bins=None, range=r, normed=None)
		figc, axc = plt.subplots(1,1)


	return avg, rms
	
