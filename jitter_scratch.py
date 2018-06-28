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

### standalone oscillators



