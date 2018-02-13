import numpy as np
from utilities import cart_to_sph, normalise
from scipy.spatial.distance import cdist


def geodesic(n_interpolation, co_ords='sph'):

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


    for i in range(n_interpolation):
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


        # INTERPOLATE AND CALCULATE NEW VERTEX LOCATIONS

        # 3D matrix with xyz co-ordnates for vertices of all faces
        v = vertices[faces_idx]

        # finding the midpoints all in one go
        midpoints = ((v + np.roll(v,1,axis=1)) / 2).reshape(-1,3)

        # # add new vertices to list
        vertices = np.append(vertices, midpoints, axis=0)

        # # find duplicate vertices
        _, idx = np.unique(vertices, axis=0, return_index=True)

        # # remove duplicates and re-sort vertices
        vertices = vertices[np.sort(idx)]

    # move vertices to unit sphere
    vertices = normalise(vertices, axis=1)

    if co_ords == 'cart':
        return vertices

    elif co_ords == 'sph':
        return cart_to_sph(vertices)



def uniform_random(npoints):
    # random sampling, uniform distribution over spherical surface

    azi = 2*np.pi * np.random.random(npoints)
    elev = np.arccos(2*np.random.random(npoints) - 1)

    return np.array([azi, elev]).T
