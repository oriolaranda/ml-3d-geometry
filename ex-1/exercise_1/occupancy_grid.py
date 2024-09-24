"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    # TODO: Implement
    coord = np.linspace(-0.5, 0.5, num=resolution)
    x, y, z = np.array(np.meshgrid(coord,coord,coord)).reshape(3,-1)
    sdf_grid = sdf_function(x, y, z).reshape((resolution,)*3)
    occ_grid = np.zeros((resolution,)*3)
    occ_grid[sdf_grid <= 0] = 1
    return occ_grid
    # ###############
