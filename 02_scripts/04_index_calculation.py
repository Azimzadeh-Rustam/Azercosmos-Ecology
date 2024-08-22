import rasterio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from sklearn.ensemble import IsolationForest
import scienceplots


FONT_SIZE = 14
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
        data = raster.read()
        return data.transpose((1, 2, 0))


def min_max_normalization(image):
    BAND_MIN_VALUE = 0.0
    BAND_MAX_VALUE = 4095.0

    return (image.astype(np.float32) - BAND_MIN_VALUE) / (BAND_MAX_VALUE - BAND_MIN_VALUE)


def split_anomalies(data):
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # select parameters
    isolation_forest = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    isolation_forest.fit(data)
    predictions = isolation_forest.predict(data)

    filtered_data = data[predictions == 1]
    anomalies = data[predictions == -1]

    return filtered_data, anomalies


def NDVI_channel(image):
    NDVI_THRESHOLD_LOW = 0.3
    NDVI_THRESHOLD_HIGH = 1.0

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    nir_channel = image[:, :, 3]

    ndvi_channel = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)
    forest_mask = np.where((ndvi_channel > NDVI_THRESHOLD_LOW) & (ndvi_channel < NDVI_THRESHOLD_HIGH), True, False)
    ndvi_forest_map = np.where(forest_mask, ndvi_channel, np.nan)

    figure = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    gs.update(wspace=0.25, hspace=0.10)

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
    ndvi_forest_filtered_data, ndvi_forest_anomalies = split_anomalies(ndvi_forest_data)
    ndvi_data_mean = np.mean(ndvi_forest_filtered_data)
    ndvi_data_median = np.median(ndvi_forest_filtered_data)
    ndvi_data_standard = np.std(ndvi_forest_filtered_data)

    bins = np.histogram_bin_edges(ndvi_forest_data, bins='scott')

    histogram_ax.set_title('Forest health histogram and boxplot')
    histogram_ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    histogram_ax.xaxis.set_major_formatter(MY_FORMATTER)
    histogram_ax.yaxis.set_major_formatter(MY_FORMATTER)
    histogram_ax.xaxis.get_offset_text().set_size(FONT_SIZE)
    histogram_ax.yaxis.get_offset_text().set_size(FONT_SIZE)
    histogram_ax.tick_params(labeltop=False, labelright=False, labelbottom=False, labelleft=True,
                             axis='both', labelsize=FONT_SIZE)
    histogram_ax.hist(ndvi_forest_filtered_data, label='Main Data', alpha=0.7, histtype='stepfilled', bins=bins,
                      color='Green')
    histogram_ax.plot([], [], ' ', label=f'Median: {ndvi_data_median:.3f}')
    histogram_ax.plot([], [], ' ', label=f'Mean: {ndvi_data_mean:.3f}')
    histogram_ax.plot([], [], ' ', label=f'Std Dev: {ndvi_data_standard:.3f}')
    histogram_ax.hist(ndvi_forest_anomalies, label='Anomalies (10%)', alpha=0.7, histtype='stepfilled', bins=bins,
                      color='Red')
    histogram_ax.legend(loc='upper left', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    boxplot_ax = figure.add_subplot(gs[1:2, 1:2])
    boxplot_ax.boxplot(ndvi_forest_data, vert=False)
    boxplot_ax.tick_params(top=True, right=False, bottom=True, left=False,
                           labeltop=False, labelright=False, labelbottom=True, labelleft=False,
                           axis='both', labelsize=FONT_SIZE)
    boxplot_ax.set_xlabel('NDVI', fontsize=FONT_SIZE)

    plt.show()
    #plt.savefig(f'../03_results/01_forests/01_NDWI/2017_NDWI.png', dpi=500)
    plt.close()


def main():
    image = read_tif('../01_src/img/2017.TIF')
    image = min_max_normalization(image)
    NDVI_channel(image)


if __name__ == '__main__':
    set_plot_style()
    main()
