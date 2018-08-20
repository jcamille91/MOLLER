import pickle
import numpy as np
from scipy.fftpack import fft, rfft
from scipy.integrate import trapz
from scipy.signal import medfilt, butter, bessel, firwin, lfilter, freqz, decimate, periodogram, welch
import matplotlib.pyplot as plt
from matplotlib import axes
ax_obj = axes.Axes

def calibrate(A, B, fsr=1.35, bits=12, ax1=None, ax2=None) :
	''' make a 2D scatter plot of A and B.
	Fit a line to it to get the slope.
	input:
	A- channel A
	B- channel B

	output:
	slope- fitted slope of the line
	'''
	# sort the numpy arrays so we can fit them.
	o = np.argsort(A)
	A = A[o]
	B = B[o]

	# 1D polynomial fit (a line)
	z = np.polyfit(A, B, 1)
	fit = np.poly1d(z)
	slope = z[0]
	offset= z[1]
	f = lambda x : offset + slope*x


	print('line has form:', offset, '+', slope, 'x')

	f1, a1 = plt.subplots(1,1)
	a1.scatter(A,B, color='blue', label='A/B')
	a1.plot(A,  f(A), color='red', label='fit')
	a1.set_xlabel('A (volts)')
	a1.set_ylabel('B (volts)')
	a1.set_title('A vs B scatter')
	a1.legend()
	f1.show()

	f2, a2 = plt.subplots(1,1)
	diff = B-(slope*A + offset)
	val, bins, pat = a2.hist(diff, bins=100)
	a2.set_xlabel('A (volts)')
	a2.set_ylabel('A - k*B + Vo (volts)')
	a2.set_title('calibrated A/B difference histogram')
	f2.show()

	# sigma = np.std(val*bins)
	print(bits, 'bit mode.' , fsr, 'volts full scale range.\n')
	print('1 bit precision = ', fsr/2**bits, 'volts')
	print('2 bit precision = ', fsr/2**(bits-1), 'volts')
	print('3 bit precision = ', fsr/2**(bits-2), 'volts\n')
	# print('A-B distribution:', sigma, 'volts sigma')

	input('press enter to finish calibration')
	return val, bins, slope, offset

def define_fir_bpf(numtap, cutoff_L, cutoff_R, fs) :
	nyq = 0.5 * fs
	normalized_cutoff_L = cutoff_L / nyq
	normalized_cutoff_R = cutoff_R / nyq
	b = firwin(numtap, (normalized_cutoff_L, normalized_cutoff_R), pass_zero=False)
	a = 1
	return b, a

def define_fir_lpf(numtap, cutoff, fs) :
	nyq = 0.5 * fs
	normalized_cutoff = cutoff / nyq
	b = firwin(numtap, normalized_cutoff)
	a = 1
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
	maxcode = 2**bits
	midpoint = 2**(bits-1)
	if raw :
		out = np.fromfile(infile, dtype=np.int16, count=-1, sep="")

	else :
		data = (np.fromfile(infile, dtype=np.int16, count=-1, sep="")-(midpoint))*fsr/(maxcode)
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

def make_data(outfile, A, fo, fs, jitter_fs, int_time, n_window) :
	''' create phase noise corrupted signal with appropriate length 
	input:
	outfile- string for .npy file storing the created signal
	A- signal amplitude
	fo- carrier freq
	fs- sampling freq
	jitter_fs- phase jitter in femtoseconds
	int_time- length of integration window in seconds
	n_window- number of integration windows
	'''
	Wo = 2*np.pi*fo
	l = int(fs*int_time)
	npt = n_window*fs
	carrier= Wo(1.0/fs*np.arange(l) + np.random.normal(loc = 0, scale = t_j*1e-15, size=len(n)))
	# array of samples. 3GSPS over a 1ms window (the helicity rate/integration time).
	n = np.linspace(0, n_window*t_flip, int(n_window*t_flip/T_s))
	#n_avg = int((len(n)-t_off)/n_sample_avg) # number of slices of the data to calculate average amplitude.

	argument = w_c*(n + np.random.normal(loc = 0, scale = t_j, size=len(n)))
	carrier = A_c*np.cos(argument)

def get_freq(data, fs, fo_nominal, ssb_bw_guess, int_time, npt2n=False, plot=False) :

	'''
	'''

	if npt2n :
		N = int(npt2n)
		binwidth = fs/N
		int_time = N/fs
	else :
		N = int(fs*int_time)	# number of samples in integration window.
		binwidth = (1/int_time) # Hz
	
	bins, v = periodogram(x=data[:N], fs=fs, nfft=None, return_onesided=True, scaling='density')
	tone_i = np.argmax(v)
	# v = rfft(x=data[:N], n=None)
	# v = np.abs(v)
	# center = int(fo_nominal*int_time)
	# left = center - int(ssb_bw_guess*int_time)
	# right = center + int(ssb_bw_guess*int_time)
	
	# tone_i = np.argmax(v[left:right]) + left
	# tone_i2 = np.argmax(v)

	tone_f = tone_i*binwidth
	# bins, power = periodogram(x=data[:N], fs=fs, nfft=None, return_onesided=True, scaling='density')
	# tone_f = np.argmax(power)*binwidth
	f_off = tone_f - fo_nominal
	print(len(data[:N]), 'data points')
	print(len(v), 'fft bins')
	print('frequency resolution = ', binwidth,'Hz\n')
	# print('calculated frequency:', tone_f, 'Hz\n')
	print('calculated frequency', tone_i2*binwidth/2)
	print('frequency offset =', f_off)



	


	if plot:
		fbins = np.linspace(0, fs/2, int(N/2))
		fig_A, ax_A = plt.subplots(1,1)

		ax_A.set_xlim(1e6, 1e10)
		ax_A.set_xscale('log')
		# ax_A.set_yscale('log')
		ax_A.set_xlabel('Frequency (Hertz)')
		ax_A.set_ylabel(r'$\frac{dBc}{Hz}$', rotation=0, fontsize=16)
		ax_A.set_title('CH A Noise Spectral Density')
		# ax_A.step(fbins, 10*np.log10(2*np.square(v)/(N*binwidth)))
		ax_A.step(fbins, 10*np.log10(v/np.max(v)))


	
	return tone_f

def xcor_spectrum(ChA, ChB, fo, fs, nbits=12, int_time=1e-3, n_window=99, plot=False) :
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
	Sdiff = (Saa + Sbb - 2*Sba)

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
	input('press enter to finish')

def spectrum(file, fo = 10e6, fs = 2850e6, int_time=1e-3, int_bw=[1,10e3], plot_en=False) :
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

def res_plot(A_avg, B_avg, ax_h=None, ax_s=None) :
	
	'''
	make histogram of the ratio of average amplitudes to show the adc resolution
	make a scatter plot of the average amplitudes of the two adc channels.
	'''

	A = A_avg/B_avg
	n=len(A)



	if ax_h == ax_obj:
		bins, vals, pat = ax_h.hist(x=A, bins=None, range=None)
		ax_h.set_xlabel('(A/B)_avg')		
		ax_h.set_ylabel('number of events')
		ax_h.set_title('(A_avg / B_avg) histogram')
	else :
		fig_h, ax_h = plt.subplots(1,1)
		bins, vals, pat = ax_h.hist(x=A, bins=None, range=None)
		ax_h.set_xlabel('samples')
		ax_h.set_ylabel('volts')
		ax_h.set_title('(A_avg / B_avg) histogram')
		fig_h.show()


	if ax_s == ax_obj:
		ax_s.scatter(x=A_avg, y=B_avg, marker='x')
		ax_s.set_xlabel('Channel A')
		ax_s.set_ylabel('Channel B')
		ax_s.set_title('(average amplitude scatter')	
	else :
		fig_s, ax_s = plt.subplots(1,1)
		ax_s.scatter(x=A_avg, y=B_avg, marker='x')
		ax_s.set_xlabel('Channel A')
		ax_s.set_ylabel('Channel B')
		ax_s.set_title('(average amplitude scatter')
		fig_s.show()
		input('press enter to close resolution plots')

	rms = np.std(A)
	print('resolution = ', rms, 'sigma')


def ddc2(ChA, ChB, fo, lpf_fc, lpf_ntaps, fs, bits, int_time, ncalc, calc_off, phase_time, nch, 
plot_en, plot_len, plot_win) :
	
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
		
	Wo = 2*np.pi*fo	
	Ts= 1/fs


	binwidth = int(1/int_time)

	# binary channels are written [A,B]_0, [A,B]_1, [A,B]_2.... etc/
	# to choose a channel pick out every other linearly indexed data point
	# if not (ChA or ChB) :
	# 	ChA, ChB = read_binary(filename=file, nbits=bits, fsr=1.35, raw=None)


		
	# this lowpass filter is for the digital downconversion
	cutoff = lpf_fc
	b0, a0 = define_fir_lpf(numtap=lpf_ntaps, cutoff=lpf_fc, fs=fs)
	# b2, a2 = define_fir_hpf(numtap=15, cutoff=hpf_fc, fs)
	w, h = freqz(b0, a0, worN=3000000)
	figf, axf = plt.subplots(1,1)
	axf.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
	axf.plot(cutoff, 0.5*np.sqrt(2), 'ko')
	axf.axvline(cutoff, color='k')
	axf.set_xlim(0, 0.5*fs)
	axf.set_title("Lowpass Filter Frequency Response")
	axf.set_xlabel('Frequency [Hz]')
	axf.grid()
	figf.show()

	# arrays to store average amplitudes and sigma (2 channels)
	avg = np.zeros((nch,ncalc))
	avg2 = np.zeros((nch,ncalc))
	rms = np.zeros(nch)

	# number of samples in 1ms window
	l = int(int_time*fs)
	npt = ncalc*l # averaging time for the phase. set to integer multiple of the integration window.

	Ch = np.array([ChA[:npt], ChB[:npt]])

	rad = Wo/fs*np.arange(l)
	I_shift= np.sin(rad) 
	Q_shift = np.cos(rad)

	I=np.zeros(len(Ch[0]))
	Q=np.zeros(len(Ch[0]))

	for k in range(nch):

		for i in range(ncalc) :

			y = (i*l)
			z = (i+1)*l

			# multiply each window (of length l) by the sin and cos modulating terms.
			I[y:z] = Ch[k][y:z]*I_shift
			Q[y:z] = Ch[k][y:z]*Q_shift

		# low pass filter frequency down-mixed data
		I_f = lfilter(b0, a0, I)
		Q_f = lfilter(b0, a0, Q)

		phase_npt = int(phase_time*fs)
		for i in range(ncalc) :
		
			c = i*l
			d = i*l + calc_off
			e = (i+1)*l

			avg_phi = np.mean(np.arctan(I_f[d:c+phase_npt]/Q_f[d:c+phase_npt]))
			
			# phase based reconstruction
			a = 2*(Q_f[d:e]*np.cos(avg_phi) + I_f[d:e]*np.sin(avg_phi))
		
			# pythagorean reconstruction
			a2 = np.hypot(I_f[d:e], Q_f[d:e])

			# average amplitude recovered from I,Q components
			avg[k][i] = np.mean(a)
			avg2[k][i] = np.mean(a2)

	
	if plot_en :

		off = plot_win*l
		plot_len = int(plot_len)
		xaxis = np.arange(plot_len)

		print('sample spacing =', Ts*1e9, 'nanoseconds')
		begin = off
		end = off + plot_len
		figA, axA = plt.subplots(1,1)
		axA.set_xlabel('samples')
		axA.set_ylabel('volts')
		axA.set_title('adc channel A raw data')
		plot(ChA[begin:end], x = xaxis, axis=axA)
		figA.show()

		# figB, axB = plt.subplots(1,1)
		# axB.set_xlabel('samples')
		# axB.set_ylabel('volts')
		# axB.set_title('adc channel B raw data')
		# plot(ChB[begin:end], x = xaxis, axis=axB)
		# figB.show()

		fig1, ax1 = plt.subplots(1,1)	
		ax1.set_xlabel('samples')
		ax1.set_ylabel('amplitude')
		ax1.set_title('I component')
		plot(I[begin:end], x = xaxis, axis=ax1)
		fig1.show()

		fig2, ax2 = plt.subplots(1,1)
		ax2.set_xlabel('samples')
		ax2.set_ylabel('amplitude')
		ax2.set_title('Q component')
		plot(Q[begin:end], x = xaxis, axis=ax2)
		fig2.show()

		fig3, ax3 = plt.subplots(1,1)
		ax3.set_xlabel('samples')
		ax3.set_ylabel('amplitude')
		ax3.set_title('I component (filtered)')
		plot(I_f[begin:end], x = xaxis, axis=ax3)
		fig3.show()

		fig4, ax4 = plt.subplots(1,1)
		ax4.set_xlabel('samples')
		ax4.set_ylabel('amplitude')
		ax4.set_title('Q component (filtered)')
		plot(Q_f[begin:end], x = xaxis, axis=ax4)
		fig4.show()
		
		# fig5, ax5 = plt.subplots(1,1)
		# ax5.set_xlabel('samples')
		# ax5.set_ylabel('amplitude')
		# ax5.set_title('Reconstructed Amplitude')
		# plot(a[begin:end], x = xaxis, axis=ax5)
		# fig5.show()


	return avg, avg2, rms

def ddc3(ChA, ChB, fo, lpf_fc, lpf_ntaps, fs, bits, int_time, ncalc, calc_off, phase_time, nch, 
plot_en, plot_len, plot_win) :

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

	Wo = 2*np.pi*fo	
	Ts= 1/fs


	binwidth = int(1/int_time)
	
	# this lowpass filter is for the digital downconversion
	cutoff = lpf_fc
	b0, a0 = define_fir_lpf(numtap=lpf_ntaps, cutoff=lpf_fc, fs=fs)
	# show the frequency magnitude response
	w, h = freqz(b0, a0, worN=3000000)
	figf, axf = plt.subplots(1,1)
	axf.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
	axf.plot(cutoff, 0.5*np.sqrt(2), 'ko')
	axf.axvline(cutoff, color='k')
	axf.set_xlim(0, 0.5*fs)
	axf.set_title("Lowpass Filter Frequency Response")
	axf.set_xlabel('Frequency [Hz]')
	axf.grid()
	figf.show()

	# arrays to store average amplitudes and sigma (2 channels)
	avg = np.zeros((nch,ncalc))
	avg2 = np.zeros((nch,ncalc))
	rms = np.zeros(nch)

	# number of samples in 1ms window
	l = int(int_time*fs)

	# total number of datapoints
	npt = ncalc*l

	Ch = np.array([ChA[:npt], ChB[:npt]])

	rad = Wo/fs*np.arange(npt)
	I_shift= np.sin(rad) 
	Q_shift = np.cos(rad)

	I=np.zeros(len(Ch[0]))
	Q=np.zeros(len(Ch[0]))

	for k in range(nch):

		I = Ch[k]*I_shift
		Q = Ch[k]*Q_shift

		I_f = lfilter(b0, a0, I)
		Q_f = lfilter(b0, a0, Q)
	
		phase_npt = int(phase_time*fs)
		phase = np.arctan(I_f[calc_off:phase_npt]/Q_f[calc_off:phase_npt])	
		avg_phase = np.mean(phase)

		for i in range(ncalc) :
			start = i*l + calc_off
			stop = (i+1)*l
			# average amplitude recovered from I,Q components

			# phase based reconstruction
			a = 2*(Q_f[start:stop]*np.cos(avg_phase) + I_f[start:stop]*np.sin(avg_phase))

			# pythagorean reconstruction
			a2 = np.hypot(I_f[start:stop], Q_f[start:stop])
			avg[k][i] = np.mean(a)
			avg2[k][i] = np.mean(a2)




	if plot_en :


		off = int(plot_win*l)
		plot_len = int(plot_len)
		xaxis = np.arange(plot_len)

		print('sample spacing =', Ts*1e9, 'nanoseconds')
		begin = off
		end = off + plot_len

		figA, axA = plt.subplots(1,1)
		axA.set_xlabel('samples')
		axA.set_ylabel('volts')
		axA.set_title('adc channel A raw data')
		plot(ChA[begin:end], x = xaxis, axis=axA)
		figA.show()

		# figB, axB = plt.subplots(1,1)
		# axB.set_xlabel('samples')
		# axB.set_ylabel('volts')
		# axB.set_title('adc channel B raw data')
		# plot(ChB[begin:end], x = xaxis, axis=axB)
		# figB.show()

		fig1, ax1 = plt.subplots(1,1)	
		ax1.set_xlabel('samples')
		ax1.set_ylabel('amplitude')
		ax1.set_title('I component')
		plot(I[begin:end], x = xaxis, axis=ax1)
		fig1.show()

		fig2, ax2 = plt.subplots(1,1)
		ax2.set_xlabel('samples')
		ax2.set_ylabel('amplitude')
		ax2.set_title('Q component')
		plot(Q[begin:end], x = xaxis, axis=ax2)
		fig2.show()

		fig3, ax3 = plt.subplots(1,1)
		ax3.set_xlabel('samples')
		ax3.set_ylabel('amplitude')
		ax3.set_title('I component (filtered)')
		plot(I_f[begin:end], x = xaxis, axis=ax3)
		fig3.show()

		fig4, ax4 = plt.subplots(1,1)
		ax4.set_xlabel('samples')
		ax4.set_ylabel('amplitude')
		ax4.set_title('Q component (filtered)')
		plot(Q_f[begin:end], x = xaxis, axis=ax4)
		fig4.show()
		
		# fig5, ax5 = plt.subplots(1,1)
		# ax5.set_xlabel('samples')
		# ax5.set_ylabel('amplitude')
		# ax5.set_title('Reconstructed Amplitude')
		# plot(a[begin:end], x = xaxis, axis=ax5)
		# fig5.show()

	return avg, avg2, rms
	
