import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import array_bounds
import rasterio
import geopandas as gpd


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def main():
    right_part_path = '../00_src/01_sentinel2/03_aoi/R20m/2017/20170728T073611.tif'

    with rasterio.open(right_part_path) as src:
        out_meta = src.meta.copy()
        print(f"Meta: {out_meta}")

        img = src.read()
        print(img.shape)
        for i in img:
            plt.imshow(i)
            plt.axis("off")
            plt.show()
            plt.close()


if __name__ == '__main__':
    main()
