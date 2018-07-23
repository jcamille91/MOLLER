# argument parse
import sys
import getopt
import argparse

# Math
import numpy as np
from scipy.signal import medfilt, butter, bessel, lfilter, freqz, decimate, periodogram, welch
from scipy.integrate import trapz
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter
# Plotting
import matplotlib.pyplot as plt
from matplotlib import axes
ax_obj = axes.Axes



def main(*args, **kwargs) :
	''' do a simulation of the effect of phase noise induced on the clock sampling signal onto the measurement
	of the RMS distribution of the amplitude'''

	parser = argparse.ArgumentParser(description="simulate sampling clock phase noise on amplitude measurement")
	parser.add_argument("n_window", type=int, help="number of integration windows for data set")
	parser.add_argument("n_sample_avg", type=int, help="number of samples each avg amplitude calculation")
	parser.add_argument("-Fs", "--samplingfreq", type=int, help="ADC sampling frequency")
	parser.add_argument("-Fc", "--carrierfreq", type=int, help="BEAM->Cavity induced carrier frequency")
	parser.add_argument("-tf", "--integrationwin", type=float, help="BEAM helicity flip rate = integration time")
	parser.add_argument("-A", "--carrieramplitude", type=float, help="Amplitude of original carrier signal", default=1.0)
	parser.add_argument("-lpfc", "--cutoff_freq", type=float, help="analog cutoff frequency for low pass filtering", default=10.0)
	parser.add_argument("-toff", "--time_off", type=int, help="offset averaging calculation by a number of samples to avoid transient effects", default=100)
	args = parser.parse_args()


	# set default values for moller CEBAF beam and TI ADC32RF45, or take optional input.
	if not args.samplingfreq : # default
		F_s = 3e9
	else :
		F_s = args.samplingfreq
	T_s = 1.0/F_s

	if not args.carrierfreq : # default
		F_c = 1.3e9
	else :
		F_c = args.carrierfreq
	w_c = 2*np.pi*F_c

	if not args.integrationwin : # default
		t_flip = 1e-3
	else :
		t_flip = args.integrationwin

	t_off = args.time_off
	A_c = args.carrieramplitude
	n_window = args.n_window
	n_sample_avg = args.n_sample_avg

	lpfc = args.cutoff_freq
	# l = int(t_flip*F_s)
	# rad = 2*np.pi*F_c/F_s*np.arange(l)
	# array of samples. 3GSPS over a 1ms window (the helicity rate/integration time).
	n = np.linspace(0, n_window*t_flip, int(n_window*t_flip/T_s))
	n_avg = int((len(n)-t_off)/n_sample_avg) # number of "n_sample_avg length" slices of the data to calculate average amplitude.

	# rectangular window cutoff approximation:  Fco = 0.44294 7/sqrt(N**2-1)
	# https://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
	avg_len = 111


	# loop over different iterations of rms_jitter, to plot the RMS value of the amplitude measurement
	# as a function of the phase noise. 

	# --------------------------------------------------------------------------------------
	# .1fs to 10fs in steps of 100.
	#tj = np.linspace(1.0e-16, 1.0e-14, 10)

	# 1fs
	# tj = np.array([1.0e-15, 10e-15, 100e-15, 1000e-15])
	# tj = np.array([1e-12, 10e-12, 20e-12])
	tj = np.array([1e-12])
	w_c2 = 2*np.pi*1300000000
	# .01fs to 10fs, 30 steps.
	# tj = np.linspace(1.0e-17, 1.0e-14, 30)
	# --------------------------------------------------------------------------------------

	# create array to store values of amplitude fluctuation
	A_rms = np.zeros(len(tj))
	A_avg = np.zeros(n_avg)

	# define butterworth lowpass filter and get its coefficients
	# coef = define_butter_lpf(cutoff=100, fs=F_s, order=5)
	cutoff = 1e9
	b, a = define_bessel_lpf(cutoff=cutoff, fs=F_s, order=7)

	# plot filter frequency response to check results are as expected

	w, h = freqz(b, a, worN=3000000)

	figf, axf = plt.subplots(1,1)
	axf.plot(0.5*F_s*w/np.pi, np.abs(h), 'b')
	axf.plot(cutoff, 0.5*np.sqrt(2), 'ko')
	axf.axvline(cutoff, color='k')
	axf.set_xlim(0, 0.5*F_s)
	axf.set_title("Lowpass Filter Frequency Response")
	axf.set_xlabel('Frequency [Hz]')
	axf.grid()
	figf.show()

	for i in range(len(tj)):

		# carrier in a 1ms window, sampled at 3GSPS with phase noise added to each time sample.
		argument = w_c*(n + np.random.normal(loc = 0, scale = tj[i], size=len(n)))
		carrier = A_c*np.cos(argument)


		# break signal into I and Q
		I = carrier*np.sin(w_c*n)
		Q = carrier*np.cos(w_c*n)
		
		# apply low-pass filter, measure the baseband component

		# I_f = digital_filter(b, a, I)
		# Q_f = digital_filter(b, a, Q)
		
		I_f = lfilter(b, a, I)
		Q_f = lfilter(b, a, Q)

		# I_f = filt(I, 'box', 111)
		# Q_f = filt(Q, 'box', 111)

		phase = np.arctan(-I_f/Q_f)	
		avg_phase = np.mean(phase)
		rms_phase = np.std(phase)

		# amplitude recovered from I,Q components
		A = 2*(Q_f*np.cos(avg_phase) - I_f*np.sin(avg_phase))
		
		# calculate average 
		for j in range(n_avg) :
			A_avg[j] = np.mean(A[j*n_sample_avg+t_off:(j+1)*n_sample_avg+t_off]) 
		A_rms[i] = np.std(A_avg)
			

	for k in range(len(tj)):

		print('{0: >5s}, {1: >5.2g}, {2: >7s}, {3: 5.9g}'.format('phase noise (fs RMS)', tj[k]*1.0e15, 'A_rms', A_rms[k]))

	print("number of points =", n_avg)
	print("window length =", n_sample_avg*T_s, "seconds")
	 ### plot the  relevant arrays, I,Q,  filtered I,Q	and reconstructed amplitude

	pltlen = 600000
	xaxis = np.arange(pltlen)*T_s*1.0e6
	
	# fig1, ax1 = plt.subplots(1,1)	
	# ax1.set_xlabel('microseconds')
	# ax1.set_ylabel('amplitude')
	# ax1.set_title('I component')
	
	# fig2, ax2 = plt.subplots(1,1)
	# ax2.set_xlabel('microseconds')
	# ax2.set_ylabel('amplitude')
	# ax2.set_title('Q component')
	
	fig3, ax3 = plt.subplots(1,1)
	ax3.set_xlabel('microseconds')
	ax3.set_ylabel('amplitude')
	ax3.set_title('I component (filtered)')
	
	fig4, ax4 = plt.subplots(1,1)
	ax4.set_xlabel('microseconds')
	ax4.set_ylabel('amplitude')
	ax4.set_title('Q component (filtered)')
	
	fig5, ax5 = plt.subplots(1,1)
	ax5.set_xlabel('microseconds')
	ax5.set_ylabel('amplitude')
	ax5.set_title('Reconstructed Amplitude')

	#a = np.sin(np.linspace(0, 7, 10000))
	
	# plot(a, npt_max=pltlen, axis=ax1)
	# plot(a, npt_max=pltlen, axis=ax2)
	# plot(a, npt_max=pltlen, axis=ax3)
	# plot(a, npt_max=pltlen, axis=ax4)
	# plot(a, npt_max=pltlen, axis=ax5)

	# plot(I, x = xaxis, axis=ax1)
	# plot(Q, x = xaxis, axis=ax2)
	plot(I_f, x = xaxis, axis=ax3)
	plot(Q_f, x = xaxis, axis=ax4)
	plot(A, x = xaxis, axis=ax5)

	# fig1.show()
	# fig2.show()
	fig3.show()
	fig4.show()
	fig5.show()

	fig6, ax6 = plt.subplots(1,1)
	ax6.set_xlim(1, 1000)
	ax6.set_ylim(1e-9, 1e-4)
	ax6.set_xscale('log')
	ax6.set_yscale('log')
	ax6.set_xlabel('rms jitter (femtoseconds)')
	ax6.set_ylabel('RMS amplitude variation')
	ax6.set_title('Resolution')
	ax6.scatter(tj*1e15, A_rms)
	fig6.show()

	input('press enter to close plots')
	plt.close('all')
	print(A_avg)
	print(A_rms)

def make_signal(A_c = 1.0, F_c = 1.3e9, F_s=3.0e9, t_j=1.0e-12, t_flip=1.0e-3, n_s_avg=300000, n_window=1) :
	''' create phase noise corrupted signal with appropriate length '''
	w_c = 2*np.pi*F_c
	T_s = 1.0/F_s
	
	# array of samples. 3GSPS over a 1ms window (the helicity rate/integration time).
	n = np.linspace(0, n_window*t_flip, int(n_window*t_flip/T_s))
	#n_avg = int((len(n)-t_off)/n_sample_avg) # number of slices of the data to calculate average amplitude.

	argument = w_c*(n + np.random.normal(loc = 0, scale = t_j, size=len(n)))
	carrier = A_c*np.cos(argument)

	return carrier


def write_binary(data, filename = '../data/test') :
	data.tofile(filename, sep="")

def read_binary(scale, filename = '../data/test') :
	'''
	this reads a binary file interpreted as series of 16bit integers, as is the case for our ADC's binary codes
	'''
	data = scale*np.fromfile(filename, dtype=np.int16, count=-1, sep="")
	return data

def calculate_jitter(ssb_pn_log, fbins, carrier) :

	'''function to calculate rms jitter (time domain expression of phase noise).
	input: 

	ssb_pn_log- single side band phase noise, relative to the carrier.
	expressed in decibels relative to the carrier in a 1 Hz bandwidth. [dBc/Hz]
	*** dBc = 10*log10(P/Pc). binwidth scaling needs to happen before the logarithm and carrier power normalization.
	fbins- linear frequency bins associated with each phase noise value provided
	carrier- linear frequency value associated with the carrier being referenced for the ssb phase noise values.

	output:
	tj_rms- rms value of the jitter, integrated from a bandwidth of the phase noise
	'''

	ssb_pn_log = np.array(ssb_pn_log)
	fbins = np.array(fbins)
	ssb_pn_lin = np.power(10, ssb_pn_log/10)
	integrated_pn = 10*np.log10(trapz(y=ssb_pn_lin, x=fbins))
	tj_rms = np.sqrt(2*10**(integrated_pn/10))/(2*np.pi*carrier)

	return tj_rms

def dbm2vpp(dbm) :

	'''convert decibels normalized to miliwatt, to volts peak-to-peak (applied to a 50ohm load)
	dbm = 10log10(P/1e-3) = 10log10(V^2/(R*1e-3)), vrms*2*sqrt(2) = vpp
	'''
	return np.sqrt(np.power(10, dbm/10)*50*1e-3*8)



def spectrum(data, Fs = 3e9, fft_len=1e-3, npt=None, scaling = 'density') :

	'''calculate the spectrum of sampled data in a given rectangular window
	input:
	data
	Fs
	fft_len
	npt ** use to override fft_len, so we can use a specific number instead of using Fs and fft_len in time.
	scaling

	output/return:
	frequency bins
	power values
	'''

	# how do we calculate the spectrum?
	# average over many windows? (this could be improved from last time with a generator expression i think)
	# the spectrum of a single window
	Ts = 1/Fs
	
	# limit FFT to each integration window, since this is the only spectrum we actually get in the 
	# physical measurement because of the helicity flip rate set at the accelerator.
	
	if npt :
		fft_npt = npt
	else :
		fft_npt = int(fft_len/Ts)
		print("number of fft points is", fft_npt, "\n")
		print("length of data is", len(data), "\n")

	#Freq_Bins, Power = welch(x=data, fs=Fs, window='hanning', nperseg=2**8, noverlap=None, nfft=fft_npt, detrend='constant', return_onesided=True, scaling='density')
	Freq_Bins, Power = periodogram(x=data, fs=Fs, window=None, nfft=fft_npt, return_onesided=True, scaling=scaling)

	return Freq_Bins, Power


def filt(d, ftype='box', box_len = '3') :
	''' apply a filer to the data 
		return the filtered data 
		input : 
		d -		n-length 1-dimensional numpy array
		ftype - string specifying type of filter to use
		f_len - integer for filter length
		'''
	if ftype =='box' :
		f = medfilt(volume=d, kernel_size=box_len)

	if ftype == 'gauss' :
		gaussian_filter()

	if ftype == 'butter' :
		# implement a butterworth lowpass filter

		butter()
		
	return f

def define_butter_lpf(cutoff, fs, order=5) :
	nyq = 0.5 * fs
	normalized_cutoff = cutoff / nyq
	b, a = butter(order, normalized_cutoff, btype='low', analog=False)
	return b, a

def define_bessel_lpf(cutoff, fs, order=5) :
	nyq = 0.5 * fs
	normalized_cutoff = cutoff / nyq
	b, a = bessel(order, normalized_cutoff, btype='low', analog=False)
	return b, a

def digital_filter(b, a, data) :
	'''takes filter coefficients (b, a) from one of the functions defining a digital filter + 
	the 1-D data to be filtered. Returns the filtered data.'''
	y = lfilter(b, a, data)
	return y





# def spectrum(d) :
# 	'''calculate and return single sided magnitude of fourier spectrum'''
# 	N = len((d))
# 	power_spectrum = np.square(np.abs(fft(d))[1:int(N/2)]/N) 	
# 	fbins = 0
# 	return power, fbins

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


if __name__ == "__main__" :
	import sys
	main(sys.argv[1:])


# function to easily calculate the spectrum
# old function averaged over multiple periods, we could make this an option with the
# helicity rate.