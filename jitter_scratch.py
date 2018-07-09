import sim as s
import numpy as np

# to convert to a carrier if not supplied, use 20log10(f1/f2) to scale the phase noise spectrum. only use this 
# for frequencies fairly close... phase noise changes over large freqency differences.

### signal generators

# ERASynth+ 
bins1 = [1e3, 10e3, 60e3, 100e3, 1e6, 10e6]
ssb1 = [-103, -110, -107, -110, -134, -150]
tj1 = s.calculate_jitter(ssb1, bins1, 3e9)

# BNC_835_4GHz
f_off2 = 20*np.log10(3e9/4e9) # data only supplied for 4 GHz
bins2 = [1e3, 10e3, 100e3, 1e6, 10e6]
ssb2 = [-104, -112, -119, -120, -128] + f_off2
tj2 = s.calculate_jitter(ssb2, bins2, 3e9)

# SRS-SG380c
f_off3 = 20*np.log10(3e9/4e9) # data only supplied for 4 GHz
bins3 = [1e3, 10e3, 100e3, 1e6, 10e6]
ssb3 = [-90, -100, -108, -120, -145] + f_off3
tj3 = s.calculate_jitter(ssb3, bins3, 3e9)

# TI LMX2582 (ADC32RF45EVM on board synth) "CLOSED LOOP PHASE NOISE" fig 6 page 10
f_off4 = 20*np.log10(3e9/5.5e9) # data only supplied for 5.5 GHz
bins4 = [1e3, 10e3, 100e3, 1e6, 10e6]
ssb4 = [-70, -86, -113, -135, -150] + f_off4
tj4 = s.calculate_jitter(ssb4, bins4, 3e9)

# TI LMX2582 (ADC32RF45EVM on board synth) "jitter calculation" fig 8 page 11
f_off5 = 20*np.log10(3e9/1.8e9) # data only supplied for 1.8 GHz
bins5 = [1e3, 10e3, 100e3, 1e6, 10e6]
ssb5 = [-110, -118, -123, -142, -156] + f_off5
tj5 = s.calculate_jitter(ssb5, bins5, 3e9)

# TI LMX2582 (ADC32RF45EVM on board synth) "jitter calculation" fig 24 page 31	
f_off6 = 20*np.log10(3e9/5.5e9) # data only supplied for 5.5 GHz
bins6 = [1e3, 10e3, 100e3, 1e6, 10e6]
ssb6 = [-100, -105, -110, -132, -150] + f_off6
tj6 = s.calculate_jitter(ssb6, bins6, 3e9)

print('ERASynth+', tj1*1e15, 'femtoseconds\n')
print('BNC_835_4GHz', tj2*1e15, 'femtoseconds\n')
print('SRS-SG380c', tj3*1e15, 'femtoseconds')
print('TI LMX2582 closed loop 5.5GHz PN (fig. 6) referred to 3GHz', tj4*1e15, 'femtoseconds\n')
print('TI LMX2582 1.8GHz PN (fig. 8) referred to 3GHz', tj5*1e15, 'femtoseconds\n')
print('TI LMX2582 5.5GHz PN (fig. 24) referred to 3GHz', tj6*1e15, 'femtoseconds\n')

### standalone oscillators



