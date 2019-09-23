import matplotlib.pyplot as plt
import numpy as np

def plot_cube(max_x,max_y,max_z, annot_x, annot_y, annot_z):
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


    # prepare some coordinates
    x, y, z = np.indices((8, 8, 8))

    # draw cuboids in the top left and bottom right corners, and a link between them
    cube1 = (x < max_x) & (y < max_y) & (z < max_z)

    # combine the objects into a single boolean array
    voxels = cube1

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[cube1] = 'white'


    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax._axis3don = False
    ax.view_init(40, 220)
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    ax.text(3.5/3*max_x, 1.5/3*max_y, max_z, annot_y)
    ax.text(1.5/3*max_x, 4/3*max_x, max_z, annot_x)
    ax.text(0, 5/3*max_y, 3/8*max_z, annot_z)

    plt.show()
