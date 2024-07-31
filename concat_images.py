import rasterio
import numpy as np

PATHS = ['src/satellite_data/R_G_B_NIR/2017/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C1.TIF',
         'src/satellite_data/R_G_B_NIR/2017/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R1C2.TIF',
         'src/satellite_data/R_G_B_NIR/2017/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C1.TIF',
         'src/satellite_data/R_G_B_NIR/2017/IMG_SPOT6_PMS_201708310709254_ORT_AKWOI-00082764_R2C2.TIF']
PATH_PAN = 'src/satellite_data/Panchromatic/2017/IMG_SPOT6_P_201708310709254_ORT_AKWOI-00083227_R1C1.TIF'
OUTPUT_PATH = 'src/img/2017.TIF'


def main():
    image = concat_images(PATHS, PATH_PAN)
    save_image(image, OUTPUT_PATH)


def read_tif(path):
    with rasterio.open(path) as raster:
        return raster.read()


def concat_images(parts_path, pan_path):
    parts = [read_tif(path) for path in parts_path]
    pan_channel = read_tif(pan_path)

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

    full_image[-1, :, :] = pan_channel[0]

    return full_image


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
