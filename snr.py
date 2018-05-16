# some basic funcitons for quickly calculating quantities relating to ADC specifications.

import numpy as np

def sum_SNR_dB(A,B):
	''' sum in quadrature two different decibel quantities, to get a new quantity in decibels.
	input:
	A, B: two quantities in decibels
	return:
	S: new sum in decibels.
	'''
	S = -20*np.log10(np.sqrt((10**(-A/20))**2 + (10**(-B/20))**2))
	return S