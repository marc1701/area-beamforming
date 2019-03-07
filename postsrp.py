import numpy as np
import utilities

# spherical peak finding
from dipy.core.sphere import Sphere
from dipy.direction import peak_directions

# machine learning
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# finding peaks in a spherical function over time
def sph_peaks_t(power_map, theta_look, phi_look, peak_thresh=.5):

    N_frames = power_map.shape[1]

    # set up sphere object for peak finding
    sph = Sphere(theta=phi_look, phi=theta_look)

    # set up output arrays for DOAs
    y_t = np.zeros((N_frames,20))
    x_t = np.zeros((N_frames,20))

    for i in range(N_frames):
        # peak finding in spherical data
        _,_,peaks = peak_directions(power_map[:,i], sph,
                                    relative_peak_threshold=peak_thresh,
                                    min_separation_angle=5)

        # save peaks to arrays
        xdirs = theta_look[peaks]
        ydirs = phi_look[peaks]
        x_t[i,0] = i
        y_t[i,0] = i
        x_t[i,1:len(xdirs)+1] += xdirs
        y_t[i,1:len(xdirs)+1] += ydirs

    # rearranging data to a useful format
    for i in range(np.min(x_t.shape)-1):

        xyi = np.append(np.append(x_t[:,[0]],x_t[:,[i+1]],1),y_t[:,[i+1]],1)

        if 'xy_t' not in locals():
            xy_t = xyi
        else:
            xy_t = np.append(xy_t, xyi, 0)

    # remove zeros
    xy_t = xy_t[np.where(xy_t[:,2] != 0)]

    return xy_t


def obj_trajectories(xy_t, eps=.4, min_samples=5, C=1e3, gamma=2):

    # set up scaler object
    cart_scaler = StandardScaler()

    # make cartesian version of spherical input data
    xy_cart = np.append(xy_t[:,[0]], utilities.sph_to_cart(xy_t[:,1:]), 1)
    # transform using scaler object
    xy_carts = cart_scaler.fit_transform(xy_cart)

    # create dbscan object and fit to data
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_carts)
    labels = db.labels_

    # support vector regression model
    svr_poly = SVR(C=C, gamma=gamma)

    # set up array for output data
    n_datapoints = labels[labels!=-1].shape[0]
    sources_out = np.zeros((n_datapoints, 4))
    last_source_idx = 0

    for label in set(labels):
        # get data relating to this source
        source = xy_carts[labels == label]
        # sort in order of time
        source = source[source[:,0].argsort()]
        # extract data to individual variables
        t = source[:,0].reshape(-1,1)
        x = source[:,1]; y = source[:,2]; z = source[:,3]

        # -1 label indicates unclustered (outlier) points
        if label != -1:
            x_poly = svr_poly.fit(t,x).predict(t).reshape(-1,1)
            y_poly = svr_poly.fit(t,y).predict(t).reshape(-1,1)
            z_poly = svr_poly.fit(t,z).predict(t).reshape(-1,1)

            xyz_poly = np.concatenate((t, x_poly, y_poly, z_poly),1)

            # scale back to actual cartesian co-ordinates
            source_inv = cart_scaler.inverse_transform(source)
            poly_inv = cart_scaler.inverse_transform(xyz_poly)
            # the 0 row of these contains the time index (frame number)

            # swap to spherical co-ordinates
            source_sph = utilities.cart_to_sph(source_inv[:,1:])
            poly_sph = utilities.cart_to_sph(poly_inv[:,1:])

            label_array = np.array([[label]] * len(poly_inv[:,[0]]))

            # prepare output list of source trajectories identified
            # sources_out format: source_n - time - azimuth - elevation
            this_source_out = np.concatenate((
                                label_array, poly_inv[:,[0]], poly_sph), 1)
            sources_out[
                last_source_idx:last_source_idx+len(
                    this_source_out), :] += this_source_out
            last_source_idx += len(this_source_out)

    # sources_out contains polynomial fit to identified sources
    # labels is a set of labels relating to the original input data xy_t
    return sources_out, labels
