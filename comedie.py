import numpy as np

def comedie_diff(soundfield):

	soundfield_cov = np.cov(soundfield)

	v_i = np.linalg.eigvals(soundfield_cov) # eigenvalues
	v_bar = np.sum(v_i)/len(v_i)

	gamma = (np.sum(np.abs(v_i - v_bar))) / v_bar
	gamma_0 = 2*(len(v_i)-1) # value of gamma in most non-diffuse case

	return np.real(1 - (gamma/gamma_0))


def diff_profile(soundfield):

	N = int(np.sqrt(len(soundfield))-1)
	profile = np.zeros((N,1))

	for n in range(N):
		soundfield_n = soundfield[:(n+2)**2,:] # remove higher order channels
		profile[n] = comedie_diff(soundfield_n)

	return np.squeeze(profile)
