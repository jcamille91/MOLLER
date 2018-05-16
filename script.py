import sim as s
import matplotlib.pyplot as plt

# this function takes 1 ms window and produces spectrum.

# next, let's observe this as a periodogram using welch's method to get the averaged spectrum of multiple windows.
# t_flip/T_s = 1e-3*3e9 = 3 million samples each window. can we do this computation without creating an array each iteration...
# we only want the result, we don't need the values of the array for future use. 
# generator expression is like list comprehension with parentheses " sum(a for i in list) "

data = s.make_signal(t_j=1e-12) # 1 picosecond of RMS jitter
bins, power = s.spectrum(data)

fig, ax = plt.subplots(1,1)

ax.set_xlim(1e6, 1e10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hertz)')
ax.set_ylabel(r'$\frac{V^2}{Hz}$', rotation=0, fontsize=16)
ax.set_title('Power Spectral Density')
ax.step(bins, power)
fig.show()

input("press enter to finish")