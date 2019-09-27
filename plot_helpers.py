import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch

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



def plot_rgb_image(img):
  # Split
  red = img[:, :, 0]
  green = img[:, :, 1]
  blue = img[:, :, 2]

  # Plot
  fig, axs = plt.subplots(2,2)

  cax_00 = axs[0,0].imshow(img)
  axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
  axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

  cax_01 = axs[0,1].imshow(red, cmap='Reds')
  fig.colorbar(cax_01, ax=axs[0,1])
  axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
  axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

  cax_10 = axs[1,0].imshow(green, cmap='Greens')
  fig.colorbar(cax_10, ax=axs[1,0])
  axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
  axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

  cax_11 = axs[1,1].imshow(blue, cmap='Blues')
  fig.colorbar(cax_11, ax=axs[1,1])
  axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
  axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
  plt.show()

  # Plot histograms
  fig, axs = plt.subplots(3, sharex=True, sharey=True)

  axs[0].hist(red.ravel(), bins=10)
  axs[0].set_title('Red')
  axs[1].hist(green.ravel(), bins=10)
  axs[1].set_title('Green')
  axs[2].hist(blue.ravel(), bins=10)
  axs[2].set_title('Blue')

  plt.show()


def french():
    blue = np.hstack([np.ones((22,22)),np.zeros((22,11))])
    green = np.hstack([np.zeros((22,11)), np.ones((22,11)),np.zeros((22,11))])
    red = np.hstack([np.zeros((22,11)), np.ones((22,22))])
    return np.moveaxis(np.array([red,green,blue]),0,-1)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 


def visualize_model(model, dataloaders, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

