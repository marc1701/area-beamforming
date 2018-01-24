import numpy as np
import scipy.special as sp

# Eigenmike capsule angles from mh Acoustics documentation
# elev_mics = (np.pi/180) * np.array([
#                            69, 90, 111, 90, 32, 55, 90, 125, 148,
#                            125, 90, 55, 21, 58, 121, 159, 69, 90,
#                            111, 90, 32, 55, 90, 125, 148, 125, 90,
#                            55, 21, 58, 122, 159]).reshape(1,-1)
# azi_mics = (np.pi/180) * np.array([
#                           0, 32, 0, 328, 0, 45, 69, 45, 0, 315, 291,
#                           315, 91, 90, 90, 89, 180, 212, 180, 148,
#                           180, 225, 249, 225, 180, 135, 111, 135,
#                           269, 270, 270, 271]).reshape(1,-1)
#
# Y_eigenmike = sph_harm_array(N, azi_mics, elev_mics)
Y_eigenmike = np.loadtxt('Y_mics.dat').view(complex)
Q = 32 # number of mic capsules

def sht( x ):
    # calculates spherical harmonic transform of
    # eigenmike space-domain signal x

    p_SH_dom = (4*np.pi / Q) * Y_eigenmike.conj().T @ x.T

    return p_SH_dom


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


def B_diag_matrix( N, k, r, r_a=None):
    # makes diagonal matrix containing b(n,kr) coefficients

    b_array = np.array([b(n, k, r, r_a)
                for n in range(N+1) for m in range(-n, n+1)])

    B = np.diag(b_array)

    return B


def B_3D( N, k, r, r_a=None ):
    # makes 3D matrix containing stack of b(n,kr) diagonals

    B = np.array([B_diag_matrix(N, k, r, r_a) for k in k])

    return B
    # B = np.array([np.diag(
    #         np.array([b(n, k, 0.042)
    #         for n in range(N+1) for m in range(-n, n+1)]))
    #     for k in k])


def sph_hankel2(n, z, derivative=False):

    h2 = (sp.spherical_jn(n, z, derivative)
          - 1j*sp.spherical_yn(n, z, derivative))

    return h2


def sph_harm_array(N, azi, elev):

    Q = np.max(elev.shape)*azi.shape[np.argmin(elev.shape)]
    # find number of angles
    # if azi and elev vectors are same orientation, Q = length of each
    # if different orientations (meshgrid) Q = product of lengths

    Y_mn = np.zeros([Q, (N+1)**2], dtype=complex)

    for i in range((N+1)**2):
        n = np.floor(np.sqrt(i))
        m = i - (n**2) - n
        # trick from ambiX paper

        Y_mn[:,i] = sp.sph_harm(m, n, azi, elev).reshape(1,-1)

    return np.array(Y_mn)


def f_to_k( f, c=343 ):
    # converts frequency in Hz to wavenumber

    k = (f / (1/(2*np.pi)) / c)

    return k


def k_to_f( k, c=343 ):
    # converts wavenumber to frequency in Hz

    f = (k*c) * (1/(2*np.pi))

    return f
