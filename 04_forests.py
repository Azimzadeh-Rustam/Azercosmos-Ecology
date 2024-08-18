import rasterio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import scienceplots


def main():
    image = read_tif('src/img/2017.TIF')
    image = min_max_normalization(image)
    NDVI_channel(image)


FONT_SIZE = 14
BAND_MIN_VALUE = 0
BAND_MAX_VALUE = 4095
NDVI_THRESHOLD_LOW = 0.3
NDVI_THRESHOLD_HIGH = 1
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-1, 1))


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-1, 1),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def min_max_normalization(image):
    num_channels = image.shape[0]
    normalized_image = image.astype(np.float32)
    for num_channel in range(num_channels):
        channel = image[num_channel]
        normalized_channel = (channel - BAND_MIN_VALUE) / (BAND_MAX_VALUE - BAND_MIN_VALUE)
        normalized_image[num_channel] = normalized_channel

    return normalized_image


def NDVI_channel(image):
    red_channel = image[0]
    nir_channel = image[3]

    ndvi_channel = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)
    forest_mask = np.where((ndvi_channel > NDVI_THRESHOLD_LOW) & (ndvi_channel < NDVI_THRESHOLD_HIGH), True, False)
    ndvi_forest_map = np.where(forest_mask, ndvi_channel, np.nan)

    figure = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    gs.update(wspace=0.25, hspace=0.25)

    forests_color_map_ax = figure.add_subplot(gs[0:2, 0:1])
    forests_color_map_ax.imshow(red_channel, cmap='Grays')
    color_map = forests_color_map_ax.imshow(ndvi_forest_map, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(color_map, ax=forests_color_map_ax, label=r'$NDVI = \frac{NIR - Red}{NIR + Red}$',
                 orientation='horizontal')
    forests_color_map_ax.set_title('NDVI color map for assessing forest health')
    forests_color_map_ax.axis('off')

    histogram_ax = figure.add_subplot(gs[0:1, 1:2])
    ndvi_forest_pattern = ndvi_channel[forest_mask]
    ndvi_forest_data = ndvi_forest_pattern.ravel()
    ndvi_data_mean = np.mean(ndvi_forest_data)
    ndvi_data_median = np.median(ndvi_forest_data)
    ndvi_data_standard = np.std(ndvi_forest_data)
    bins = np.histogram_bin_edges(ndvi_forest_data, bins='scott')

    histogram_ax.set_title('Forest health histogram')
    histogram_ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    histogram_ax.xaxis.set_major_formatter(MY_FORMATTER)
    histogram_ax.yaxis.set_major_formatter(MY_FORMATTER)
    histogram_ax.xaxis.get_offset_text().set_size(FONT_SIZE)
    histogram_ax.yaxis.get_offset_text().set_size(FONT_SIZE)
    histogram_ax.tick_params(labeltop=False, labelright=False, labelbottom=False, labelleft=True,
                             axis='both', labelsize=FONT_SIZE)
    histogram_ax.hist(ndvi_forest_data, label=f'Forest health ({NDVI_THRESHOLD_LOW} < NDVI < {NDVI_THRESHOLD_HIGH})',
                      alpha=0.7, histtype='stepfilled', bins=bins, color='Green')
    histogram_ax.plot([], [], ' ', label=f'Median: {ndvi_data_median:.3f}')
    histogram_ax.plot([], [], ' ', label=f'Mean: {ndvi_data_mean:.3f}')
    histogram_ax.plot([], [], ' ', label=f'Std Dev: {ndvi_data_standard:.3f}')
    histogram_ax.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    boxplot_ax = figure.add_subplot(gs[1:2, 1:2])
    boxplot_ax.boxplot(ndvi_forest_data, vert=False)
    boxplot_ax.tick_params(top=True, right=False, bottom=True, left=False,
                           labeltop=False, labelright=False, labelbottom=True, labelleft=False,
                           axis='both', labelsize=FONT_SIZE)
    boxplot_ax.set_xlabel('NDVI', fontsize=FONT_SIZE)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    set_plot_style()
    main()
