import rasterio
import numpy as np
import torch
from skimage.transform import resize

PATHS = ['../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C1.TIF',
         '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C2.TIF',
         '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C1.TIF',
         '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C2.TIF']
OUTPUT_PATH = '../01_src/03_images/2017-s.TIF'


def main():
    image = concat_parts(PATHS)
    image = min_max_normalization(image, band_min_value=0.0, band_max_value=4095.0)
    image = resample_image(image, current_resolution=1.5, target_resolution=5.0)
    #image = spectral_reconstruction(image)
    save_image(image, OUTPUT_PATH)


def min_max_normalization(image, band_min_value, band_max_value):
    return (image - band_min_value) / (band_max_value - band_min_value)


def read_tif(path):
    with rasterio.open(path) as raster:
        data = raster.read()
        data = data.transpose((1, 2, 0))
        return data.astype(np.float32)


def concat_parts(parts_path):
    parts = [read_tif(path) for path in parts_path]

    parts_info = [part.shape for part in parts]
    num_channels = parts_info[0][0] + 1
    total_height = max(parts_info[0][1] + parts_info[2][1], parts_info[1][1] + parts_info[3][1])
    total_width = max(parts_info[0][2] + parts_info[1][2], parts_info[2][2] + parts_info[3][2])
    full_image = np.zeros((num_channels, total_height, total_width), dtype=parts[0].dtype)

    y_offset_R2 = parts_info[0][1]
    x_offset_C2 = parts_info[0][2]
    offsets = [(0, 0), (0, x_offset_C2), (y_offset_R2, 0), (y_offset_R2, x_offset_C2)]

    for part, (y_offset, x_offset) in zip(parts, offsets):
        part_height, part_width = part.shape[1], part.shape[2]
        full_image[:num_channels - 1, y_offset:y_offset + part_height, x_offset:x_offset + part_width] = part

    return full_image


def resample_image(image, current_resolution, target_resolution):
    scale_factor = current_resolution / target_resolution

    initial_height, initial_width = image.shape[0], image.shape[1]

    new_height = int(initial_height * scale_factor)
    new_width = int(initial_width * scale_factor)

    return resize(image, (new_height, new_width), order=1, mode='constant', cval=0, anti_aliasing=True,
                  preserve_range=True)


def save_image(image_data, output_path):
    with rasterio.open(PATHS[0]) as src:
        meta = src.meta.copy()
    meta.update({
        'count': image_data.shape[0],
        'height': image_data.shape[1],
        'width': image_data.shape[2],
        'dtype': image_data.dtype
    })
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(image_data)


if __name__ == '__main__':
    main()
