import rasterio
from matplotlib import pyplot as plt
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import numpy as np
import geopandas as gpd


def transform_geojson_to_raster_crs(geojson_path, raster_crs):
    gdf = gpd.read_file(geojson_path)
    gdf_transformed = gdf.to_crs(raster_crs)
    return gdf_transformed


def merge_parts(parts_paths):
    with rasterio.open(parts_paths[0]) as part_1, \
        rasterio.open(parts_paths[1]) as part_2, \
        rasterio.open(parts_paths[2]) as part_3, \
        rasterio.open(parts_paths[3]) as part_4:

        parts = [part_1, part_2, part_3, part_4]
        mosaic, mosaic_transform = merge(parts)

        mosaic = mosaic.astype(np.float32)

        b_g_r_nir_channels_order = [2, 1, 0, 3]
        mosaic = mosaic[b_g_r_nir_channels_order, ...]

        out_meta = part_1.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'count': mosaic.shape[0],
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'dtype': str(mosaic.dtype),
            'crs': part_1.crs,
            'transform': mosaic_transform,
            'BIGTIFF': 'YES',
            'compress': 'lzw'
        })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(mosaic)
            memory_dataset.close()
        return memory_raster.open()


def rescale_image(image: rasterio.io.DatasetReader, current_resolution: float, target_resolution: float) -> rasterio.io.DatasetReader:
    scale_factor = current_resolution / target_resolution

    num_channels = image.count
    initial_height = image.height
    initial_width = image.width

    new_height = int(initial_height * scale_factor)
    new_width = int(initial_width * scale_factor)

    data = image.read(
        out_shape=(num_channels, new_height, new_width),
        resampling=Resampling.bilinear
    )

    out_meta = image.meta.copy()
    original_transform = out_meta['transform']
    new_transform = rasterio.Affine(
        original_transform.a / scale_factor,
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e / scale_factor,
        original_transform.f
    )

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


def crop_raster(image: rasterio.io.DatasetReader, geojson_gdf: gpd.GeoDataFrame) -> rasterio.io.DatasetReader:
    cropped_image, cropped_transform = mask(image, geojson_gdf.geometry, crop=True, nodata=np.nan)

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
            memory_dataset.close()
        return memory_raster.open()


def min_max_normalization(raster: rasterio.io.DatasetReader, band_min_value: float, band_max_value: float) -> rasterio.io.DatasetReader:
    image = raster.read()
    normalized_image = (image.astype(np.float32) - band_min_value) / (band_max_value - band_min_value)

    out_meta = raster.meta.copy()
    out_meta.update({
        'dtype': str(normalized_image.dtype)
    })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(normalized_image)
            memory_dataset.close()
        return memory_raster.open()


def save_raster(image: rasterio.io.DatasetReader, output_path: str):
    with rasterio.open(output_path, 'w', **image.meta) as output_raster:
        output_raster.write(image.read())


def main():
    PATHS = [
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762_R1C1.TIF',
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762_R1C2.TIF',
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762_R2C1.TIF',
        '../00_src/02_azersky/01_raw_data/R_G_B_NIR/DIM_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_PMS_001_A/IMG_SPOT6_PMS_202307150717126_ORT_AKWOI-00082762_R2C2.TIF']
    OUTPUT_PATH = '../00_src/02_azersky/02_aoi/R20m/202307150717126.tif'
    GEOJSON_PATH = '../00_src/AOI.geojson'

    with rasterio.open(PATHS[0]) as src:
        transformed_geojson = transform_geojson_to_raster_crs(GEOJSON_PATH, src.crs)

    full_raster = merge_parts(PATHS)
    rescaled_image = rescale_image(full_raster, current_resolution=1.5, target_resolution=20.0)
    cropped_area = crop_raster(rescaled_image, transformed_geojson)
    #image = atmospheric_correction(image)
    normalized_raster = min_max_normalization(cropped_area, band_min_value=0.0, band_max_value=4095.0)
    #image = restore_spectra(image)
    save_raster(normalized_raster, OUTPUT_PATH)


if __name__ == '__main__':
    main()
