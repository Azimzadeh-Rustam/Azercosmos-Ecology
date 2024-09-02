import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
import numpy as np


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as raster:
        return raster.read(1)


def main():
    right_part_path = '../00_src/01_sentinel2/02_tiles/2022/T39TUG_20220816T073619.tif'
    left_part_path = '../00_src/01_sentinel2/02_tiles/2022/T39TTG_20220816T073619.tif'

    with (rasterio.open(left_part_path) as left_part,
          rasterio.open(right_part_path) as right_part):

        mosaic, out_trans = merge([left_part, right_part])

        print(mosaic.shape)
        plt.imshow(mosaic[5])
        plt.show()


        # Сохранение результата
        #out_meta = src1.meta.copy()
        #out_meta.update({"driver": "GTiff",
        #                 "height": mosaic.shape[1],
        #                 "width": mosaic.shape[2],
        #                 "transform": out_trans,
        #                 "compress": "lzw"})

        #with rasterio.open('mosaic_output.tif', 'w', **out_meta) as dest:
        #    dest.write(mosaic)


if __name__ == '__main__':
    main()
