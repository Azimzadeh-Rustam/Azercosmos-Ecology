import matplotlib.pyplot as plt
import rasterio
import numpy as np
import torch
import RRDBNet_arch as arch


PATCH_SIZE = 256
DEVICE = torch.device('cuda')


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as raster:
        return raster.read()


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image - band_min_value) / (band_max_value - band_min_value)


def visualise_rgb(image):
    red_channel, green_channel, blue_channel = image[2], image[1], image[0]
    true_color_composite = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    p_low, p_high = np.nanpercentile(true_color_composite, [0.5, 99.5])
    clipped_image = np.clip(true_color_composite, p_low, p_high)
    normalized_image = min_max_normalization(clipped_image, p_low, p_high)

    plt.imshow(normalized_image, vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()




def main():
    path = '../00_src/01_sentinel2/03_aoi/R20m/20240805T073619.tif'
    image = read_raster(path)
    visualise_rgb(image)

    #esrgan_model = load_esrgan_model()
    #image = increase_spatial_resolution(image=image, model=esrgan_model)

    #print(image.shape)
    #for i in image:
    #    plt.imshow(i)
    #    plt.show()
    #    plt.close()


if __name__ == '__main__':
    main()
