import rasterio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scienceplots

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
    NDVI_channel(image)


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def NDVI_channel(image):
    red_channel = image[0]
    nir_channel = image[3]
    ndvi_channel = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    color_map = ax1.imshow(ndvi_channel, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(color_map, ax=ax1)
    ax1.set_title('NDVI color map')
    ax1.axis('off')

    ax2 = axes[1]
    ndvi_data = ndvi_channel[(ndvi_channel > 0.3) & (ndvi_channel < 1)]
    ndvi_data = ndvi_data.ravel()
    ndvi_data_mean = ndvi_data.mean()
    ndvi_data_standard = ndvi_data.std()
    bins = np.histogram_bin_edges(ndvi_data, bins='scott')

    ax2.set_title('')
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_xlabel(r'$NDVI = \frac{NIR - Red}{NIR + Red}$', fontsize=10)
    ax2.xaxis.set_major_formatter(my_formatter)
    ax2.yaxis.set_major_formatter(my_formatter)
    ax2.xaxis.get_offset_text().set_size(FONT_SIZE)
    ax2.yaxis.get_offset_text().set_size(FONT_SIZE)
    ax2.tick_params(axis='both', labelsize=FONT_SIZE)
    ax2.hist(ndvi_data, label='Forest health (0.3 > NDVI > 1)', alpha=0.7, histtype='stepfilled', bins=bins, color='Green')
    ax2.plot([], [], ' ', label=f'Mean: {ndvi_data_mean:.3f}')
    ax2.plot([], [], ' ', label=f'Std Dev: {ndvi_data_standard:.3f}')
    ax2.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
