import rasterio
import numpy as np
from skimage.transform import resize
import torch
import RRDBNet_arch as arch
from rasterio.transform import Affine


PATCH_SIZE = 256
DEVICE = torch.device('cuda')


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as raster:
        return raster.read(1)


def unify_bands(bands: list) -> list:
    random_channel_num = 0
    nir_channel_num = 4
    height, width = bands[random_channel_num].shape

    bands[nir_channel_num] = resize(bands[nir_channel_num], (height, width), order=1, anti_aliasing=True,
                                    preserve_range=True)

    return bands


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image.astype(np.float32) - band_min_value) / (band_max_value - band_min_value)


def load_esrgan_model() -> torch.nn.Module:
    ESRGAN_MODEL_PATH = '../00_src/RRDB_ESRGAN_x4.pth'

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(ESRGAN_MODEL_PATH), strict=True)
    model = model.eval()
    return model.to(DEVICE)


def pad_to_multiple(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape[1], image.shape[2]

    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    paddings = [(0, 0), (0, pad_height), (0, pad_width)]

    return np.pad(image, paddings, mode='constant', constant_values=np.nan)


def increase_spatial_resolution(image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    num_channels, initial_height, initial_width = image.shape

    scale_factor = 4
    scaled_height = initial_height * scale_factor
    scaled_width = initial_width * scale_factor
    scaled_image = np.zeros((num_channels, scaled_height, scaled_width), dtype=np.float32)

    for channel_index in range(2, num_channels):
        if channel_index == 2:
            rgb_indices = [2, 1, 0]
            true_color_composite = image[rgb_indices, ...]
        else:
            channel = image[channel_index]
            zeros_channel = np.zeros_like(channel)
            true_color_composite = np.stack([channel, zeros_channel, zeros_channel], axis=0)

        true_color_tensor = torch.from_numpy(true_color_composite).to(dtype=torch.float32)
        true_color_tensor = true_color_tensor.unsqueeze(0)
        low_resolution_input = true_color_tensor.to(DEVICE)
        with torch.no_grad():
            high_resolution_output = model(low_resolution_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        if channel_index == 2:
            bgr_indices = [2, 1, 0]
            scaled_image[:3, ...] = high_resolution_output[bgr_indices, ...]
        else:
            scaled_image[channel_index] = high_resolution_output[0]

    return scaled_image


def increase_spatial_resolution_by_patches(image: np.ndarray, model: torch.nn.Module, patch_size: int) -> np.ndarray:
    num_channels, initial_height, initial_width = image.shape

    multiple_image = pad_to_multiple(image, patch_size)
    _, multiple_height, multiple_width = multiple_image.shape

    scale_factor = 4
    scaled_multiple_height = multiple_height * scale_factor
    scaled_multiple_width = multiple_width * scale_factor
    scaled_multiple_image = np.zeros((num_channels, scaled_multiple_height, scaled_multiple_width), dtype=np.float32)

    for channel_index in range(2, num_channels):
        if channel_index == 2:
            true_color_composite = multiple_image[[2, 1, 0]]
        else:
            channel = multiple_image[channel_index]
            zeros_channel = np.zeros_like(channel)
            true_color_composite = np.stack([channel, zeros_channel, zeros_channel], axis=0)

        for start_y in range(0, multiple_height, patch_size):
            for start_x in range(0, multiple_width, patch_size):
                end_y = start_y + patch_size
                end_x = start_x + patch_size
                patch = true_color_composite[:, start_y:end_y, start_x:end_x]

                patch_input = torch.from_numpy(patch).to(dtype=torch.float32)
                patch_input = patch_input.unsqueeze(0)
                patch_input = patch_input.to(DEVICE)
                with torch.no_grad():
                    scaled_patch = model(patch_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()

                scaled_start_y = start_y * scale_factor
                scaled_start_x = start_x * scale_factor
                scaled_end_y = scaled_start_y + patch_size * scale_factor
                scaled_end_x = scaled_start_x + patch_size * scale_factor
                if channel_index == 2:
                    scaled_multiple_image[:3, scaled_start_y:scaled_end_y, scaled_start_x:scaled_end_x] = scaled_patch[[2, 1, 0]]
                else:
                    scaled_multiple_image[channel_index, scaled_start_y:scaled_end_y, scaled_start_x:scaled_end_x] = scaled_patch[0]

    scaled_height = initial_height * scale_factor
    scaled_width = initial_width * scale_factor

    return scaled_multiple_image[:, :scaled_height, :scaled_width]


def get_metadata(path: str) -> dict:
    with rasterio.open(path) as raster:
        return raster.meta.copy()


def save_raster(image: np.ndarray, meta: dict, output_path: str) -> None:
    initial_height, initial_width = meta['height'], meta['width']
    num_channels, new_height, new_width = image.shape

    x_scale = initial_width / new_width
    y_scale = initial_height / new_height

    new_transform = meta['transform'] * Affine.scale(x_scale, y_scale)

    meta.update({
        'driver': 'GTiff',
        'count': num_channels,
        'height': new_height,
        'width': new_width,
        'dtype': str(image.dtype),
        'crs': rasterio.crs.CRS.from_epsg(32639),
        'transform': new_transform,
        "compress": "lzw"
    })

    with rasterio.open(output_path, 'w', **meta) as output_raster:
        output_raster.write(image)


def main():
    band_paths = [
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R20m/T39TUG_20170728T073611_B02_20m.jp2', # Blue
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R20m/T39TUG_20170728T073611_B03_20m.jp2', # Green
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R20m/T39TUG_20170728T073611_B04_20m.jp2', # Red
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R20m/T39TUG_20170728T073611_B05_20m.jp2', # Red Edge 1
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R10m/T39TUG_20170728T073611_B08_10m.jp2', # NIR
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R20m/T39TUG_20170728T073611_B8A_20m.jp2', # Red Edge 2
        '../00_src/01_sentinel2/01_raw_data/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TUG_20231007T092611.SAFE/GRANULE/L2A_T39TUG_A010957_20170728T074308/IMG_DATA/R20m/T39TUG_20170728T073611_B11_20m.jp2' # SWIR 1
    ]
    output_path = '../00_src/01_sentinel2/02_tiles/R20m/2017/T39TUG_20170728T073611.tif'

    bands = [read_raster(band_path) for band_path in band_paths]
    bands = unify_bands(bands)
    image = np.stack(bands, axis=0)
    image = min_max_normalization(image, band_min_value=0.0, band_max_value=65535.0) # 16 bit

    esrgan_model = load_esrgan_model()
    image = increase_spatial_resolution(image=image, model=esrgan_model)
    #image = increase_spatial_resolution_by_patches(image=image, model=esrgan_model, patch_size=PATCH_SIZE)

    meta_data = get_metadata(band_paths[0])
    save_raster(image, meta=meta_data, output_path=output_path)


if __name__ == '__main__':
    main()
