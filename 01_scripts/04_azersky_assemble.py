import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import numpy as np
import geopandas as gpd


def merge_parts(parts_paths):
    with rasterio.open(parts_paths[0]) as part_1, \
        rasterio.open(parts_paths[1]) as part_2, \
        rasterio.open(parts_paths[2]) as part_3, \
        rasterio.open(parts_paths[3]) as part_4:

        parts = [part_1, part_2, part_3, part_4]
        mosaic, mosaic_transform = merge(parts)

        b_g_r_nir_channels_order = [2, 1, 0, 3]
        mosaic = mosaic[b_g_r_nir_channels_order, ...]

        out_meta = part_1.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'count': mosaic.shape[0],
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'crs': rasterio.crs.CRS.from_epsg(32639),
            'transform': mosaic_transform,
            'compress': 'lzw'
        })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(mosaic)
            memory_dataset.close() # new
        return memory_raster.open()


def rescale_image(image: rasterio.io.DatasetReader, current_resolution: float, target_resolution: float) -> rasterio.io.DatasetReader:
    scale_factor = current_resolution / target_resolution

    initial_height = image.height
    initial_width = image.width
    num_channels = image.count

    new_height = int(initial_height * scale_factor)
    new_width = int(initial_width * scale_factor)

    new_transform = image.transform * image.transform.scale(
        (initial_width / new_width),
        (initial_height / new_height)
    )

    data = image.read(
        out_shape=(num_channels, new_height, new_width),
        resampling=Resampling.bilinear
    )

    out_meta = image.meta.copy()
    out_meta.update({
        'count': num_channels,
        "height": new_height,
        "width": new_width,
        "transform": new_transform
    })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(data)
            memory_dataset.close()
            return memory_raster.open()


def crop_raster(image: rasterio.io.DatasetReader, geojson_path: str) -> rasterio.io.DatasetReader:
    shapes = gpd.read_file(geojson_path)
    cropped_image, cropped_transform = mask(image, shapes.geometry, crop=True, nodata=np.nan)

    out_meta = image.meta.copy()
    out_meta.update({
        "height": cropped_image.shape[1],
        "width": cropped_image.shape[2],
        "transform": cropped_transform,
        "nodata": np.nan
    })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(cropped_image)
            memory_dataset.close() # new
        return memory_raster.open()


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image.astype(np.float32) - band_min_value) / (band_max_value - band_min_value)


def save_raster(image: rasterio.io.DatasetReader, output_path: str):
    with rasterio.open(output_path, 'w', **image.meta) as output_raster:
        output_raster.write(image.read())


def main():
    PATHS = [
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C1.TIF',
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C2.TIF',
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C1.TIF',
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C2.TIF']
    OUTPUT_PATH = '../00_src/02_azersky/02_aoi/R5m/201708310709254.tif'
    GEOJSON_PATH = '../00_src/AOI.geojson'

    image = merge_parts(PATHS)
    rescaled_image = rescale_image(image, current_resolution=1.5, target_resolution=5.0)
    cropped_area = crop_raster(rescaled_image, GEOJSON_PATH)
    image = atmospheric_correction(image)
    image = min_max_normalization(image, band_min_value=0.0, band_max_value=4095.0)
    image = restore_spectra(image)
    save_raster(image, OUTPUT_PATH)


if __name__ == '__main__':
    main()
