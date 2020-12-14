import numpy as np
from utilities import cart_to_sph, sph_to_cart, normalise
from scipy.spatial.distance import cdist

# golden ratio
R = (1 + np.sqrt(5)) / 2

def regular(N, co_ords='sph'):

    # find N for each dimension, resulting in smallest possible
    # whole number of points above input N
    N = int(np.ceil(np.sqrt(N)))

    # meshgrid of points
    x, y = np.meshgrid(np.linspace(0, 2*np.pi, N),#[:-1],
                       np.linspace(0, np.pi, N))#[1:-1])
    # [1:-1] avoids duplicate points at poles and wraparound

    # reshape into a list of points
    points = np.stack((x, y)).reshape(2,-1).T

    if co_ords == 'cart':
        return sph_to_cart(points)

    elif co_ords == 'sph':
        return np.array(points)


def geodesic(N_interp, return_points='vertices', co_ords='sph'):

    # DEFINE INITIAL ICOSAHEDRON
    # using orthogonal rectangle method
    # http://sinestesia.co/blog/tutorials/python-icospheres/

    vertices = np.array([[-1,R,0],
                         [1,R,0],
                         [-1,-R,0],
                         [1,-R,0],

                         [0,-1,R],
                         [0,1,R],
                         [0,-1,-R],
                         [0,1,-R],

                         [R,0,-1],
                         [R,0,1],
                         [-R,0,-1],
                         [-R,0,1]])

    for n in range(N_interp + 1):
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


        # if N_interp has been reached, break off here
        if n == N_interp:

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


def random(N, co_ords='sph'):
    # random sampling, uniform distribution over spherical surface

    theta = 2*np.pi * np.random.random(N)
    phi = np.arccos(2*np.random.random(N) - 1)

    if co_ords == 'cart':
        return sph_to_cart(np.array([theta, phi]).T)

    elif co_ords == 'sph':
        return np.array([theta, phi]).T


def fibonacci(N, co_ords='sph'):
    # quasi-regular sampling using fibonacci spiral

    i = np.arange(N)

    theta = 2*np.pi*i/R
    # arccos as we use spherical co-ordinates rather than lat-lon
    phi = np.arccos(-(2*i/N-1))

    if co_ords == 'cart':
        return sph_to_cart(np.array([theta, phi]).T)

    elif co_ords == 'sph':
        return np.array([theta, phi]).T % (2*np.pi)
