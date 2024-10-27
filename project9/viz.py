import numpy as np
import matplotlib.pyplot as plt


def raster_plot(spikes, title=None, figsize=(12, 6)):
    plt.figure(figsize=figsize)

    # Transpose so that sample is MxT
    spikes = spikes.T

    # Find row/col indices where there are spikes
    r_inds, c_inds = np.nonzero(spikes)
    # Loop through and draw a vertical "tick" at each spot
    for i in range(len(r_inds)):
        plt.plot([c_inds[i], c_inds[i]], [r_inds[i], r_inds[i]+1], c='k', marker='|')

    plt.xlabel('Time (steps)')
    plt.ylabel('Feature')
    plt.ylim(0, spikes.shape[0])
    plt.title(title)
    plt.show()


def draw_grid_image(data, n_cols, n_rows, sample_dims=(28, 28), title=''):
    # reshape each sample into format: (N, n_rows, n_cols)
    data = data.reshape((-1, sample_dims[0], sample_dims[1]))
    # select only the samples that fit into the grid
    data = data[np.arange(n_rows*n_cols)]

    # make an empty canvas on which we place the individual images
    canvas = np.zeros([sample_dims[0]*n_rows, sample_dims[1]*n_cols])
    # (r,c) becomes the top-left corner
    for r in range(n_rows):
        for c in range(n_cols):
            ind = r*n_cols + c
            canvas[r*sample_dims[0]:(r+1)*sample_dims[0], c*sample_dims[1]:(c+1)*sample_dims[1]] = data[ind]

    # For live updating: clear current plot in figure
    # plt.clf()

    max = np.max(np.abs(canvas))
    im = plt.imshow(canvas, cmap='bwr', vmin=-max, vmax=max)
    fig = plt.gcf()
    fig.colorbar(im, ticks=[np.min(canvas), 0, np.max(canvas)])

    if title is not None:
        plt.title(title)

    plt.axis('off')


def plot_voltage(v, title='LIF neurons'):
    # For live updating: clear current plot in figure
    plt.clf()

    plt.plot(v)

    plt.xlabel('Time (msec)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)

    plt.tight_layout()


def plot_voltage_stack(excit_v, inhib_v, titles=('Excit neurons', 'Inhib neurons')):
    '''NOT PROVIDED'''
    # For live updating: clear current plot in figure
    plt.clf()

    for r in range(2):
        ax = plt.subplot(2, 1, r+1)

        if r == 0:
            ax.plot(excit_v)
        else:
            ax.plot(inhib_v)

        ax.set_title(titles[r])
        ax.set_ylabel('Voltage (mV)')

        if r == 1:
            ax.set_xlabel('Time (msec)')

    plt.tight_layout()


def visualize_sample_animation(x_enc, yi, sample_spatial_dims=(28, 28), figsize=(8, 8)):
    '''NOT PROVIDED
    '''
    fig = plt.figure(figsize=figsize)

    # Reshape back into 28x28 images
    x_enc = x_enc.reshape([len(x_enc), sample_spatial_dims[0], sample_spatial_dims[1]])

    plt.axis('off')
    for t in range(len(x_enc)):
        plt.clf()
        plt.imshow(x_enc[t])
        plt.title(yi)

        fig.canvas.draw()

    plt.show()
