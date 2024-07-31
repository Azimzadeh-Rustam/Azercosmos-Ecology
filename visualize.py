import rasterio
import numpy as np
from skimage.transform import resize
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scienceplots


CHANNELS = ['Red', 'Green', 'Blue', 'Near-Infrared', 'Panchromatic']
NUM_CHANNELS = len(CHANNELS)
BAND_MAX_VALUE = 4095
FIGURE_NUM_COLUMNS = 3
FIGURE_NUM_ROWS = 2
FONT_SIZE = 10

plt.style.use(['science', 'notebook', 'grid'])
matplotlib.rcParams.update({'font.size': FONT_SIZE})
matplotlib.rcParams["axes.formatter.limits"] = (-1, 1)
matplotlib.rcParams['axes.formatter.useoffset'] = True
matplotlib.rcParams['axes.formatter.offset_threshold'] = 1
my_formatter = ScalarFormatter(useMathText=True)
my_formatter.set_scientific(True)
my_formatter.set_powerlimits((-1, 1))


def main():
    image = read_tif('src/img/2017.TIF')
    visualize_channels(image)
    channel_histograms(image)


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def compress_image(image, factor):
    height_compressed = image.shape[0] // factor
    width_compressed = image.shape[1] // factor

    image_compressed = resize(image, (height_compressed, width_compressed), mode='reflect', anti_aliasing=False)
    #image_compressed -= image_compressed.min()
    image_compressed /= image_compressed.max()
    return image_compressed


def visualize_channels(image):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))

    red_compressed = compress_image(image[0], 10)
    green_compressed = compress_image(image[1], 10)
    blue_compressed = compress_image(image[2], 10)
    nir_compressed = compress_image(image[3], 10)
    panchromatic_compressed = compress_image(image[4], 10)
    rgb_compressed = np.dstack((red_compressed, green_compressed, blue_compressed))

    axis1 = axes[0][0]
    red_canvas = axis1.imshow(red_compressed, cmap='Reds')
    axis1.set_title('Red chanel')
    axis1.axis('off')
    figure.colorbar(red_canvas, ax=axis1)

    axis2 = axes[0][1]
    green_canvas = axis2.imshow(green_compressed, cmap='Greens')
    axis2.set_title('Green chanel')
    axis2.axis('off')
    figure.colorbar(green_canvas, ax=axis2)

    axis3 = axes[0][2]
    blue_canvas = axis3.imshow(blue_compressed, cmap='Blues')
    axis3.set_title('Blue chanel')
    axis3.axis('off')
    figure.colorbar(blue_canvas, ax=axis3)

    axis4 = axes[1][0]
    nir_canvas = axis4.imshow(nir_compressed, cmap='magma')
    axis4.set_title('Near-Infrared chanel')
    axis4.axis('off')
    figure.colorbar(nir_canvas, ax=axis4)

    axis5 = axes[1][1]
    panchromatic_canvas = axis5.imshow(panchromatic_compressed, cmap='grey')
    axis5.set_title('Panchromatic chanel')
    axis5.axis('off')
    figure.colorbar(panchromatic_canvas, ax=axis5)

    axis6 = axes[1][2]
    axis6.imshow(rgb_compressed)
    axis6.set_title('Visible light')
    axis6.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def channel_histograms(image):
    colors = ['red', 'green', 'blue', 'purple', 'gray']
    num_channel = 0

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row in range(FIGURE_NUM_ROWS):
        for column in range(FIGURE_NUM_COLUMNS):
            if num_channel < NUM_CHANNELS:
                channel = image[num_channel]
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


if __name__ == '__main__':
    main()
