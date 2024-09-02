import rasterio
import numpy as np
from skimage.transform import resize
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scienceplots


PATH = 'src/img/2017.TIF'
CHANNELS = ['Red', 'Green', 'Blue', 'Near-Infrared', 'Panchromatic']
NUM_CHANNELS = len(CHANNELS)
FIGURE_NUM_COLUMNS = 3
FIGURE_NUM_ROWS = 2
FONT_SIZE = 10
COMPRESSION_FACTOR = 10

plt.style.use(['science', 'notebook', 'grid'])
matplotlib.rcParams.update({'font.size': FONT_SIZE})
matplotlib.rcParams["axes.formatter.limits"] = (-1, 1)
matplotlib.rcParams['axes.formatter.useoffset'] = True
matplotlib.rcParams['axes.formatter.offset_threshold'] = 1
my_formatter = ScalarFormatter(useMathText=True)
my_formatter.set_scientific(True)
my_formatter.set_powerlimits((-1, 1))


def read_tif(path):
    with rasterio.open(path) as raster:
        data = raster.read()
        return data.transpose((1, 2, 0))


def min_max_normalization(image):
    BAND_MIN_VALUE = 0.0
    BAND_MAX_VALUE = 4095.0

    return (image.astype(np.float32) - BAND_MIN_VALUE) / (BAND_MAX_VALUE - BAND_MIN_VALUE)


def enhance_exposure(image):
    GAMMA = 0.8
    image = np.power(image, GAMMA)
    return image


def visualize_rgb(image):
    true_color_composite = image[:, :, :3]
    #image = enhance_exposure(image)

    plt.figure(figsize=(10, 10))
    plt.imshow(true_color_composite, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_channels(image):
    CMAPS = ['Reds', 'Greens', 'Blues', 'gray', 'gray']

    figure, axes = plt.subplots(FIGURE_NUM_ROWS, FIGURE_NUM_COLUMNS, figsize=(14, 8))

    for num_channel, axis in enumerate(axes.flatten()):
        axis.axis('off')
        if num_channel < NUM_CHANNELS:
            channel = image[:, :, num_channel]
            color_map = axis.imshow(channel, vmin=0, vmax=1, cmap=CMAPS[num_channel])
            axis.set_title(f'{CHANNELS[num_channel]} channel')
            figure.colorbar(color_map, ax=axis)
        else:
            true_color_composite = image[:, :, :3]
            axis.imshow(true_color_composite, vmin=0, vmax=1)
            axis.set_title('Visible light')

    plt.tight_layout()
    plt.show()
    plt.close()


def channel_histograms(image):
    colors = ['red', 'green', 'blue', 'purple', 'gray']
    num_channel = 0

    figure, axes = plt.subplots(FIGURE_NUM_ROWS, FIGURE_NUM_COLUMNS, figsize=(14, 8))

    for row in range(FIGURE_NUM_ROWS):
        for column in range(FIGURE_NUM_COLUMNS):
            if num_channel < NUM_CHANNELS:
                channel = image[:, :, num_channel]
                channel_nonzero_values = channel[channel > 0]
                channel_data = channel_nonzero_values.ravel()
                channel_data_mean = channel_data.mean()
                channel_data_standard = channel_data.std()
                bins = np.histogram_bin_edges(channel_data, bins='scott')

                ax = axes[row, column]
                ax.set_title(f'{CHANNELS[num_channel]} channel')
                ax.set_ylabel('Frequency', fontsize=10)
                ax.xaxis.set_major_formatter(my_formatter)
                ax.yaxis.set_major_formatter(my_formatter)
                ax.xaxis.get_offset_text().set_size(FONT_SIZE)
                ax.yaxis.get_offset_text().set_size(FONT_SIZE)
                ax.tick_params(axis='both', labelsize=FONT_SIZE)
                ax.hist(channel_data, histtype='bar', bins=bins, color=colors[num_channel])
                ax.plot([], [], ' ', label=f'Mean: {channel_data_mean:.3f}')
                ax.plot([], [], ' ', label=f'Std Dev: {channel_data_standard:.3f}')
                ax.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

                num_channel += 1
            else:
                axes[row, column].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    image = read_tif(PATH)
    image = min_max_normalization(image)
    visualize_rgb(image)
    visualize_channels(image)
    channel_histograms(image)


if __name__ == '__main__':
    main()
