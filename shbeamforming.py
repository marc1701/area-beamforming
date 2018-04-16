import numpy as np
import scipy.special as sp
import spherical_sampling
from utilities import *

def sph_harm_array(N, theta, phi, sh_type='real'):

    # Q = np.max(phi.shape)*theta.shape[np.argmin(phi.shape)]
    # find number of angles
    if type(theta) == float:
        Q = 1
    else:
        Q = len(theta)

    Y_mn = np.zeros([Q, (N+1)**2], dtype=complex)

    for i in range((N+1)**2):

        # trick from ambiX paper
        n = np.floor(np.sqrt(i))
        m = i - (n**2) - n

        Y_mn[:,i] = sp.sph_harm(m, n, theta, phi).reshape(1,-1)

    if sh_type == 'real':
        Y_mn = np.real((C(N) @ Y_mn.T).T)

    return np.array(Y_mn)


def g0(N_sh):
    return np.sqrt( (2*N_sh + 1) / (N_sh+1)**2 )

def d_minimum_sidelobe(N_sh, n_sh):
    # equation from Delikaris-Manias 2016
    return (g0(N_sh) * (sp.gamma(N_sh+1) * sp.gamma(N_sh+2) /
                        sp.gamma(N_sh+1+n_sh) * sp.gamma(N_sh+3+n_sh)))


# rotations of spherical functions - equations from Rafaely2015 Chpt. 1.6
# exponents of e here to positive rather than negative j
# this matches with results from other formulations I have found
# usually only using real part anyway
def wgnr_D(n, m, mp, alpha, beta, gamma):
    return np.e**(-1j*mp*alpha) * wgnr_d(n, m, mp, beta) * np.e**(-1j*m*gamma)


def wgnr_d(n, m, mp, beta):

    mu = np.abs(mp-m)
    vu = np.abs(mp+m)
    s = n-(mu+vu)/2

    zeta = 1 if m >= mp else (-1)**int(m-mp)

    return ( zeta *
        np.sqrt( (factorial(s)*factorial(s+mu+vu)) /
                  (factorial(s+mu)*factorial(s+vu)) ) *
        np.sin(beta/2)**mu * np.cos(beta/2)**vu *
        eval_jacobi(s, mu, vu, np.cos(beta)) )


def wgnr_D_mat(N, alpha, beta, gamma):
    D = OrderedDict()

    for n in range(N+1):
        indices = rotation_indices(n)
        D[n] = np.array([wgnr_D(nmmp[0], nmmp[1], nmmp[2], alpha, beta, gamma)
                   for nmmp in indices]).reshape(2*n+1,2*n+1)

    return block_diag(*[n for _, n in D.items()])


def rotation_indices(n):
    return np.array([[n,m,mp] for n in range(n,n+1)
    for m in range(-n, n+1) for mp in range(-n, n+1)])


def C(N):

    C = OrderedDict()

    for n in range(N+1):
        C[n] = c(n)

    return block_diag(*[x for _, x in C.items()])


def c(N): # complex/real transform matrix
    indices = rotation_indices(N)

    C = np.zeros((2*N+1)**2, dtype=complex)

    for i, nmmp in enumerate(indices):
        n, m, mp = nmmp[0], nmmp[1], nmmp[2]

        if abs(m) != abs(mp):
            C[i]  = 0
        elif m - mp == 0: # same sign
            if m == 0: # both 0
                C[i] = np.sqrt(2)
            elif m < 0: # both negative
                C[i] = 1j
            else: # both positive
                C[i] = (int(-1)**int(m))
        elif m - mp > 0: # mp negative
            C[i] = 1
        elif m - mp < 0: # mp positive
            C[i] = -1j*(int(-1)**int(m))

    C *= 1/(np.sqrt(2))

    return C.reshape(2*N+1, 2*N+1)


def delta(N, alpha, beta, gamma):
    # real version of Wigner-D rotation matrix
    D = wgnr_D_mat(N, alpha, beta, gamma)

    return np.real(np.conj(C(N)) @ D @ C(N).T)


def rotate_SH(beampattern, theta, phi, psi):
    # rotates the Z axis to the direction specified
    # helps if your beampattern begins facing straight up at 0, 0
    # this essentially wraps wigner_D function in easier to understand terms
    N = int(np.sqrt(len(beampattern)) - 1)
    return delta(N, psi, -phi, theta) @ beampattern


#########################################################################
### EVERYTHING BELOW THIS LINE NEEDED ONLY WHEN WORKING WITH
### RAW EIGENMIKE AUDIO OUTPUT - (SEMI) DEPRECATED

### non-optimal rigid sphere compensation (division) resulting in
### amplification of microphone self-noise at low frequencies
### in high-order channels

### also requires use of unweildy 3D matrix for rigid sphere compensation
### this could possibly be rectified with reformulation of SHT function

# Eigenmike capsule angles from mh Acoustics documentation
# phi_mics = np.radians(np.array([
#                            69, 90, 111, 90, 32, 55, 90, 125, 148,
#                            125, 90, 55, 21, 58, 121, 159, 69, 90,
#                            111, 90, 32, 55, 90, 125, 148, 125, 90,
#                            55, 21, 58, 122, 159]).reshape(1,-1))
# theta_mics = np.radians(np.array([
#                           0, 32, 0, 328, 0, 45, 69, 45, 0, 315, 291,
#                           315, 91, 90, 90, 89, 180, 212, 180, 148,
#                           180, 225, 249, 225, 180, 135, 111, 135,
#                           269, 270, 270, 271]).reshape(1,-1))
#
# Y_eigenmike = sph_harm_array(N, theta_mics, phi_mics)
Y_eigenmike = np.loadtxt('Y_mics.dat').view(complex)
Q = 32 # number of mic capsules

def sht( x, N ):
    # calculates spherical harmonic transform of
    # eigenmike space-domain signal x up to order N

    p_nm = (4*np.pi / Q) * Y_eigenmike[:, :(N+1)**2].conj().T @ x.T

    return p_nm


def srp_fft( x, fs, N_fft=None ):
    # calculate fft for use in SRP algorithm
    # zeros noise-dominated lower bands in higher-order SH channels
    # removes bands above spatial nyquist (8 kHz for Eigenmike)

    p_nm_k = np.fft.fft(x, axis=1, n=N_fft)
    # take FFT of incoming time-SH-domain frame

    if not N_fft:
        N_fft = p_nm_k.shape[1]
        # infer FFT length if not specified

    f_vals = np.linspace(0, fs/2, N_fft//2)
    k_vals = f_to_k(f_vals)
    # calculate frequency and wavenumber bin values

    cutoff_2nd = np.searchsorted(f_vals, [400])[0]
    cutoff_3rd = np.searchsorted(f_vals, [1000])[0]
    cutoff_4th = np.searchsorted(f_vals, [1800])[0]
    spatial_nyquist = np.searchsorted(f_vals, [8000])[0]+1
    # find bins for low cutoff frequencies and spatial nyquist
    # figures for these listed in Eigenmike documentation

    p_nm_k[4:9,:cutoff_2nd] = 0
    p_nm_k[9:16,:cutoff_3rd] = 0
    p_nm_k[16:,:cutoff_4th] = 0
    p_nm_k = p_nm_k[:, 1:spatial_nyquist]
    # zero frequency bands as calculated above and trim
    # spatially-aliased high frequency bands

    k_vals = k_vals[1:spatial_nyquist]
    # trim list of wavenumbers to match

    return p_nm_k, k_vals


def b( n, k, r, r_a=None ):
    # n = SH order
    # k = wavenumber
    # r = mic position radius
    # r_a = rigid sphere radius

    if r_a == None:
        r_a = r # r_a = r with eigenmike

    b = (4 * np.pi * 1j**n
    * (sp.spherical_jn(n, k*r) -
    sp.spherical_jn(n, k*r_a, True) / sph_hankel2(n, k*r_a, True)
    * sph_hankel2(n, k*r)))

    return b


def B_diag_matrix( N, k, r, beampattern='pwd', r_a=None):
    # makes diagonal matrix containing b(n,kr) coefficients

    b_array = np.array([b(n, k, r, r_a)
                for n in range(N+1) for m in range(-n, n+1)])

    if beampattern == 'pwd':
        d = 1

    elif beampattern == 'min_sidelobe':
        d = np.array([d_minimum_sidelobe(N, n)
                for n in range(N+1) for m in range(-n, n+1)])

    # apply weights for selected beam pattern and diagonalify
    B = np.diag(d/b_array)

    return B


def B_3D( N, k, r, beampattern='pwd', r_a=None ):
    # makes 3D matrix containing stack of b(n,kr) diagonals

    B = np.array([B_diag_matrix(N, k, r, beampattern, r_a) for k in k])

    return B
    # B = np.array([np.diag(
    #         np.array([b(n, k, 0.042)
    #         for n in range(N+1) for m in range(-n, n+1)]))

    #     for k in k])


def sph_hankel2(n, z, derivative=False):

    h2 = (sp.spherical_jn(n, z, derivative)
          - 1j*sp.spherical_yn(n, z, derivative))

    return h2
