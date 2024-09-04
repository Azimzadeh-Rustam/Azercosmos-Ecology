import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.mask import mask
import numpy as np
from skimage.transform import resize


def merge_parts(parts_paths):
    with rasterio.open(parts_paths[0]) as part_1, \
        rasterio.open(parts_paths[1]) as part_2, \
        rasterio.open(parts_paths[2]) as part_3, \
        rasterio.open(parts_paths[3]) as part_4:

        parts = [part_1, part_2, part_3, part_4]
        mosaic, mosaic_transform = merge(parts)

        out_meta = part_1.meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform
        })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(mosaic)
        return memory_raster.open()


def resample_image(image, current_resolution, target_resolution):
    scale_factor = current_resolution / target_resolution

    initial_height, initial_width = image.shape[0], image.shape[1]

    new_height = int(initial_height * scale_factor)
    new_width = int(initial_width * scale_factor)

    return resize(image, (new_height, new_width), order=1, mode='constant', cval=0, anti_aliasing=True,
                  preserve_range=True)


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image.astype(np.float32) - band_min_value) / (band_max_value - band_min_value)


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


def main():
    PATHS = [
        '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C1.TIF',
        '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C2.TIF',
        '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C1.TIF',
        '../01_src/01_azercosmos_images/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C2.TIF']
    OUTPUT_PATH = '../01_src/03_images/2017-s.tif'

    image = merge_parts(PATHS)
    image = resample_image(image, current_resolution=1.5, target_resolution=5.0)
    image = min_max_normalization(image, band_min_value=0.0, band_max_value=4095.0)
    #image = spectral_reconstruction(image)
    save_image(image, OUTPUT_PATH)


if __name__ == '__main__':
    main()
