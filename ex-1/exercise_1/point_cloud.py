"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    triangle_areas = []
    for face in faces:
        [v_1,v_2,v_3] = vertices[face]
        triangle_area = 0.5*np.linalg.norm(np.cross(v_2-v_1,v_3-v_1))
        triangle_areas.append(triangle_area)
    triangle_areas = np.array(triangle_areas)
    probabilities = triangle_areas/triangle_areas.sum()
    sampled_points_triangle = np.random.choice(range(len(triangle_areas)), size=n_points,p=probabilities)
    sampled_points = []
    for face in sampled_points_triangle:
        [A,B,C] = vertices[faces[face]]
        [r1,r2] = np.random.rand(2)
        u = 1 - np.sqrt(r1)
        v = np.sqrt(r1)*(1-r2)
        w = np.sqrt(r1)*r2
        point = u*A + v*B + w*C
        sampled_points.append(point)
    return np.array(sampled_points)
    # ###############
