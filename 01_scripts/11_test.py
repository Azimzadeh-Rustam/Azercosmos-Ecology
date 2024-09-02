import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import array_bounds


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def main():
    #image = read_tif('../00_src/01_sentinel2/02_tiles/2022/T39TTG_20220816T073619.tif')
    with rasterio.open('../00_src/01_sentinel2/02_tiles/2022/T39TTG_20220816T073619.tif') as raster:
        transform = raster.transform
        height, width = raster.height, raster.width
        dataset_crs = raster.crs
        left, bottom, right, top = array_bounds(height, width, transform)

    print("Координаты углов изображения:")
    print("Левая граница (Left):", left)
    print("Нижняя граница (Bottom):", bottom)
    print("Правая граница (Right):", right)
    print("Верхняя граница (Top):", top)


if __name__ == '__main__':
    main()
