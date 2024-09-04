import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.mask import mask
import geopandas as gpd


def merge_parts(path_1: str, path_2: str) -> rasterio.io.DatasetReader:
    with (rasterio.open(path_1) as part_1,
          rasterio.open(path_2) as part_2):

        mosaic, mosaic_trans = merge([part_1, part_2])

        out_meta = part_1.meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_trans
        })

    with MemoryFile() as memory_raster:
        with memory_raster.open(**out_meta) as memory_dataset:
            memory_dataset.write(mosaic)
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
        return memory_raster.open()


def save_raster(image: rasterio.io.DatasetReader, output_path: str):
    with rasterio.open(output_path, 'w', **image.meta) as output_raster:
        output_raster.write(image.read())


def main():
    right_part_path = '../00_src/01_sentinel2/02_tiles/R20m/2017/T39TTG_20170728T073611.tif'
    left_part_path = '../00_src/01_sentinel2/02_tiles/R20m/2017/T39TUG_20170728T073611.tif'
    geojson_path = '../00_src/AOI.geojson'
    output_path = '../00_src/01_sentinel2/03_aoi/R20m/2017/20170728T073611.tif'

    full_area = merge_parts(left_part_path, right_part_path)
    cropped_area = crop_raster(full_area, geojson_path)
    save_raster(cropped_area, output_path)


if __name__ == '__main__':
    main()
