import rasterio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scienceplots

CHANNELS = ['Blue', 'Green', 'Red', 'Red Edge 1', 'Near-Infrared', 'Red Edge 2', 'SWIR']
NUM_CHANNELS = len(CHANNELS)
FIGURE_NUM_COLUMNS = 3
FIGURE_NUM_ROWS = 3
FONT_SIZE = 14
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-2, 2))

def set_plot_style() -> None:
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-2, 2),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def main() -> None:
    PATH = '../00_src/01_sentinel2/03_aoi/R20m/20180703T073611.tif'

    image = read_raster(PATH)
    visualise_rgb(image)
    visualize_channels(image)
    channel_histograms(image)


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as raster:
        return raster.read()


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image - band_min_value) / (band_max_value - band_min_value)


def normalize_with_clipping(image: np.ndarray) -> np.ndarray:
    p_low, p_high = np.nanpercentile(image, [0.5, 99.5])
    clipped_image = np.clip(image, p_low, p_high)
    return min_max_normalization(clipped_image, p_low, p_high)


def visualise_rgb(image: np.ndarray) -> None:
    red_channel, green_channel, blue_channel = image[2], image[1], image[0]
    true_color_composite = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    normalized_image = normalize_with_clipping(true_color_composite)

    plt.imshow(normalized_image, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_channels(image):
    CMAPS = ['Blues', 'Greens', 'Reds', 'gray', 'gray', 'gray', 'gray']

    figure, axes = plt.subplots(FIGURE_NUM_ROWS, FIGURE_NUM_COLUMNS, figsize=(14, 14))

    for num_channel, axis in enumerate(axes.flatten()):
        axis.axis('off')
        if num_channel < NUM_CHANNELS:
            channel = image[num_channel]
            normalized_channel = normalize_with_clipping(channel)
            color_map = axis.imshow(normalized_channel, vmin=0, vmax=1, cmap=CMAPS[num_channel])
            axis.set_title(f'{CHANNELS[num_channel]} channel')
            figure.colorbar(color_map, ax=axis)

    plt.tight_layout()
    plt.show()
    plt.close()


def channel_histograms(image):
    colors = ['blue', 'green', 'red', 'purple', 'gray', 'gray', 'gray']
    num_channel = 0

    figure, axes = plt.subplots(FIGURE_NUM_ROWS, FIGURE_NUM_COLUMNS, figsize=(14, 14))

    for row in range(FIGURE_NUM_ROWS):
        for column in range(FIGURE_NUM_COLUMNS):
            if num_channel < NUM_CHANNELS:
                channel = image[num_channel]
                channel_data = channel.flatten()
                channel_data = channel_data[~np.isnan(channel_data)]

                channel_median = np.median(channel_data)
                channel_mean = np.mean(channel_data)
                channel_std = np.std(channel_data)
                channel_max = np.max(channel_data)
                channel_min = np.min(channel_data)

                bins = np.histogram_bin_edges(channel_data, bins='scott')

                ax = axes[row, column]
                ax.set_title(f'{CHANNELS[num_channel]} channel')
                ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
                ax.xaxis.set_major_formatter(MY_FORMATTER)
                ax.yaxis.set_major_formatter(MY_FORMATTER)
                ax.xaxis.get_offset_text().set_size(FONT_SIZE)
                ax.yaxis.get_offset_text().set_size(FONT_SIZE)
                ax.tick_params(axis='both', labelsize=FONT_SIZE)
                ax.hist(channel_data, histtype='stepfilled', alpha=0.7, bins=bins, color=colors[num_channel], range=(0, 1))
                ax.set_xlim(-0.05, 1.05)
                ax.plot([], [], ' ', label=f'Median: {channel_median:.3f}')
                ax.plot([], [], ' ', label=f'Mean: {channel_mean:.3f}')
                ax.plot([], [], ' ', label=f'Std Dev: {channel_std:.3f}')
                ax.plot([], [], ' ', label=f'Min: {channel_min:.3f}')
                ax.plot([], [], ' ', label=f'Max: {channel_max:.3f}')
                ax.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

                num_channel += 1
            else:
                axes[row, column].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    set_plot_style()
    main()
