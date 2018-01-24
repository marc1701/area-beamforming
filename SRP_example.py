import shbeamforming as shb
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import soundfile as sf

audio, fs = sf.read('noise.wav')
audio = audio[fs*5:fs*6,:]
# trim audio to a one-second clip (includes static noise)

ks = shb.f_to_k(fs)
# work out wavenumber 'sampling frequency'

N_fft = 1024
p_space_dom = np.fft.fft(audio[:N_fft,:], axis=0, n=N_fft)
# calculate FFT of frame
# this becomes space-frequency domain signals e.g. P(k)

k_vals = np.linspace(ks/1024, ks/2, N//2)
# create vector with wavenumber values for FFT bins


# np.where(k_to_f(k_vals)==9000)
# k_to_f(k_vals[191]) # find location of useful/desired frequency

# plt.plot(np.multiply(np.hanning(N), audio[:N,:]))
# hanning window might be useful to use if calculating SRP in consecutive frames


p_SH_dom = shb.sht(p_space_dom)
# discrete spherical harmonic transform (SHT)

N = 4 # Eigenmike is 4th order

azi_look = np.linspace(0, 2*np.pi, 60).reshape(1,-1)
elev_look = np.linspace(0, np.pi, 60).reshape(-1,1)
# these arrays are rotated relative to one another to make axes

Y_look = shb.sph_harm_array(N, azi_look, elev_look)
# # calculate spherical harmonics for look directions

# k_vals[35:255] # wavenumber indeces

B_mat = np.linalg.inv(shb.B_3D(N, k_vals[35:255], 0.042))
# setting up scattering (B) matrices with custom B_3D function

p_sh = np.rollaxis(np.expand_dims(p_SH_dom[:,35:255],2),1,0)
# expand dims of SH domain signals for correct multiplication
# roll axis of p_sh for correct multiplication

W_N = np.squeeze(Y_look @ B_mat @ p_sh)
# @ = matrix multiplication
# sequential multiplication through the 3D stack of B diagonal matrices
# squeeze W_N to remove resultant unneeded dimension

beta_zk = np.array([1]*(255-35)) # weighting function
W_N = abs(W_N).T @ beta_zk # this does the same as the below line
# but allows us to potentially change the weighting contour
# W_N = abs(W_N).sum(0) # sum across all k vals

plt.contourf(W_N.reshape(60,60))
# plt.savefig('abs.pdf')
