"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    with open(path, "w") as f:
        for v in vertices:
            f.write("v "+" ".join(map(str, v))+"\n")
        for face in faces:
            f.write("f "+" ".join(map(str, face))+"\n")
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    with open(path, "w") as f:
        for p in pointcloud:
            f.write("v "+" ".join(map(str, p))+"\n")
    # ###############
