import rasterio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import scienceplots

FONT_SIZE = 14
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-2, 2))


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-2, 2),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def detect_outliers(data):
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    anomalies = np.array([])
    filtered_data = data

    threshold = 1.5

    while True:
        Q1 = np.quantile(filtered_data, 0.25)
        Q3 = np.quantile(filtered_data, 0.75)
        IQR = Q3 - Q1

        upper_bound = Q3 + threshold * IQR
        lower_bound = Q1 - threshold * IQR

        is_anomaly = (filtered_data < lower_bound) | (filtered_data > upper_bound)
        new_anomalies = filtered_data[is_anomaly]

        if len(new_anomalies) == 0:
            break

        anomalies = np.concatenate((anomalies, new_anomalies))
        filtered_data = filtered_data[~is_anomaly]

    return filtered_data, anomalies


def index_channel(image, save_path):
    THRESHOLD_LOW = 0.0
    THRESHOLD_HIGH = 1.0

    blue_channel = image[0]
    green_channel = image[1]
    red_channel = image[2]
    red_edge_1_channel = image[3]
    nir_channel = image[4]
    red_edge_4_channel = image[5]
    swir1_channel = image[6]

    background = red_channel

    mndwi_channel = (green_channel - swir1_channel) / (green_channel + swir1_channel + 1e-10)
    sea_mask = np.where((mndwi_channel > THRESHOLD_LOW) & (mndwi_channel < THRESHOLD_HIGH), True, False)

    ndci_channel = (red_edge_1_channel - red_channel) / (red_edge_1_channel + red_channel + 1e-10)

    # Calculate forest area
    #spatial_resolution = 20
    #pixel_area_m2 = spatial_resolution ** 2
    #forest_pixel_count = np.nansum(forests_mask)
    #forest_area_m2 = forest_pixel_count * pixel_area_m2
    #forest_area_km2 = int(forest_area_m2 / 1e6)

    area_map = np.where(sea_mask, ndci_channel, np.nan)

    figure = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2)
    gs.update(wspace=0.2, hspace=0.05)

    # ================== COLOR MAP ==================
    area_color_map_ax = figure.add_subplot(gs[0:2, 0:1])
    area_color_map_ax.imshow(background, cmap='Grays')
    color_map = area_color_map_ax.imshow(area_map, cmap='RdYlBu_r') #, vmin=-1, vmax=1
    plt.colorbar(color_map, ax=area_color_map_ax, label=r'$NDCI\,=\,\frac{Red\,Edge\,1\,(B5)\,-\,Red\,(B4)}{Red\,Edge\,1\,(B5)\,+\,Red\,(B4)}$',
                 orientation='horizontal', shrink=.75)
    area_color_map_ax.set_title('Normalized Difference Chlorophyll Index (NDCI)\ncolor map')
    area_color_map_ax.axis('off')

    # ================== HISTOGRAM ==================
    histogram_ax = figure.add_subplot(gs[0:1, 1:2])

    index_area_pattern = ndci_channel[sea_mask] #change
    index_area_data = index_area_pattern.ravel()
    index_area_filtered_data, index_area_anomalies = detect_outliers(index_area_data)
    index_data_mean = np.mean(index_area_filtered_data)
    index_data_median = np.median(index_area_filtered_data)
    index_data_standard = np.std(index_area_filtered_data)

    bins = np.histogram_bin_edges(index_area_data, bins='scott')

    histogram_ax.set_title('Histogram and Boxplot\nof Algal Bloom Content in the Sea')
    histogram_ax.set_ylabel('Frequency', fontsize=FONT_SIZE)
    histogram_ax.xaxis.set_major_formatter(MY_FORMATTER)
    histogram_ax.yaxis.set_major_formatter(MY_FORMATTER)
    histogram_ax.xaxis.get_offset_text().set_size(FONT_SIZE)
    histogram_ax.yaxis.get_offset_text().set_size(FONT_SIZE)
    histogram_ax.tick_params(labeltop=False, labelright=False, labelbottom=False, labelleft=True,
                             axis='both', labelsize=FONT_SIZE)
    histogram_ax.hist(index_area_filtered_data, label='Filtered Data', alpha=0.7, histtype='stepfilled', bins=bins,
                      color='Green')
    histogram_ax.plot([], [], ' ', label=f'Median: {index_data_median:.3f}')
    histogram_ax.plot([], [], ' ', label=f'Mean: {index_data_mean:.3f}')
    histogram_ax.plot([], [], ' ', label=f'Std Dev: {index_data_standard:.3f}')
    #histogram_ax.plot([], [], ' ', label=f'Forest area: {forest_area_km2} $km^2$')
    if len(index_area_anomalies) != 0:
        histogram_ax.hist(index_area_anomalies, label='Anomalies', alpha=0.7, histtype='stepfilled', bins=bins,
                          color='Red')
    histogram_ax.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    # ================== BOXPLOT ==================
    boxplot_ax = figure.add_subplot(gs[1:2, 1:2])
    if len(index_area_anomalies) != 0:
        boxplot_ax.boxplot([index_area_data, index_area_filtered_data], vert=False, widths=0.3)
        boxplot_ax.tick_params(top=True, right=True, bottom=True, left=True,
                               labeltop=False, labelright=False, labelbottom=True, labelleft=True,
                               axis='both', labelsize=FONT_SIZE)
        boxplot_ax.set_yticklabels(['Original Data', 'Filtered Data'], rotation=45, fontsize=FONT_SIZE)
    else:
        boxplot_ax.boxplot(index_area_data, vert=False)
        boxplot_ax.tick_params(top=True, right=False, bottom=True, left=False,
                               labeltop=False, labelright=False, labelbottom=True, labelleft=False,
                               axis='both', labelsize=FONT_SIZE)
    boxplot_ax.set_xlabel('NDCI value', fontsize=FONT_SIZE)

    #plt.show()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    IMAGE_PATH = '../00_src/01_sentinel2/04_aoi/20240805T073619.tif'
    OUTPUT_PATH = '../02_results/05_sea_indices/01_NDCI/20240805T073619.png'
    image = read_tif(IMAGE_PATH)
    index_channel(image, OUTPUT_PATH)


if __name__ == '__main__':
    set_plot_style()
    main()
