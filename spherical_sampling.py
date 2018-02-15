import numpy as np
from utilities import cart_to_sph, normalise
from scipy.spatial.distance import cdist


def geodesic(n_interpolation, return_points='vertices', co_ords='sph'):

    # DEFINE INITIAL ICOSAHEDRON
    # using orthogonal rectangle method
    # http://sinestesia.co/blog/tutorials/python-icospheres/

    t = (1 + np.sqrt(5)) / 2

    vertices = np.array([[-1,t,0],
                         [1,t,0],
                         [-1,-t,0],
                         [1,-t,0],

                         [0,-1,t],
                         [0,1,t],
                         [0,-1,-t],
                         [0,1,-t],

                         [t,0,-1],
                         [t,0,1],
                         [-t,0,-1],
                         [-t,0,1]])

    for n in range(n_interpolation + 1):
        # CALCULATION OF SIDES

        # find euclidian distances between all points -
        # gives us a matrix of distances
        euclid_dists = cdist(vertices, vertices)

        # find list of adjacent vertices
        sides_idx = np.where(
            euclid_dists == np.min(euclid_dists[euclid_dists > 0]))

        # concatenate output locations into one array
        sides_idx = np.concatenate(
            (sides_idx[0].reshape(-1,1), sides_idx[1].reshape(-1,1)), axis=1)

        # remove duplicate sides_idx (there are many)
        _, idx = np.unique(np.sort(sides_idx), axis=0, return_index=True)
        sides_idx = sides_idx[idx]


        # CALCULATION OF FACES

        # set up empty array
        faces_idx = np.array([], dtype=int)

        for i in np.unique(sides_idx[:,0]):
            # extract sides_idx related to each vertex
            a = sides_idx[np.where(sides_idx[:,0] == i),1]

            for j in a:
                for l in j:
                    # find 3rd adjacent vertices common to both points
                    b = sides_idx[np.where(sides_idx[:,0] == l), 1]
                    intersect = np.intersect1d(a,b).reshape(-1,1)

                    for m in intersect:
                        # add faces_idx to array
                        faces_idx = np.append(faces_idx, np.array([i,l,m]))

        # output is a 1D list, so we need to reshape it
        faces_idx = faces_idx.reshape(-1,3)

        # 3D matrix with xyz co-ordnates for vertices of all faces
        v = vertices[faces_idx]


        # if n_interpolation has been reached, break off here
        if n == n_interpolation:

            # FIND MIDPOINTS OF EACH FACE
            # this finds the dodecahedron-like relation to the
            # icosahedron at different interpolation levels
            facepoints = v.sum(axis=1)/3

            if return_points == 'faces':
                vertices = facepoints

            elif return_points == 'both':
                vertices = np.append(vertices, facepoints, axis=0)

            # move vertices to unit sphere
            vertices = normalise(vertices, axis=1)

            if co_ords == 'cart':
                return vertices

            elif co_ords == 'sph':
                return cart_to_sph(vertices)


        # INTERPOLATE AND CALCULATE NEW VERTEX LOCATIONS

        # finding the midpoints all in one go
        midpoints = ((v + np.roll(v,1,axis=1)) / 2).reshape(-1,3)

        # # add new vertices to list
        vertices = np.append(vertices, midpoints, axis=0)

        # # find duplicate vertices
        _, idx = np.unique(vertices, axis=0, return_index=True)

        # # remove duplicates and re-sort vertices
        vertices = vertices[np.sort(idx)]



def uniform_random(N):
    # random sampling, uniform distribution over spherical surface

    azi = 2*np.pi * np.random.random(N)
    elev = np.arccos(2*np.random.random(N) - 1)

    return np.array([azi, elev]).T



def fibonacci(N):
    # quasi-regular sampling using fibonacci spiral

    # golden ratio
    T = (1 + np.sqrt(5)) / 2

    i = np.arange(N)

    theta = 2*np.pi*i/T
    # arccos as we use spherical co-ordinates rather than lat-lon
    phi = np.arccos(2*i/N-1)

    return np.array([theta,phi]).T
