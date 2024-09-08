import rasterio
import numpy as np
from skimage.transform import resize
import torch
import RRDBNet_arch as arch
from rasterio.crs import CRS

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


def increase_spatial_resolution(image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    num_channels, initial_height, initial_width = image.shape

    scale_factor = 4
    scaled_height = initial_height * scale_factor
    scaled_width = initial_width * scale_factor
    scaled_image = np.zeros((num_channels, scaled_height, scaled_width), dtype=np.float32)

    for flag in range(2, num_channels):
        if flag == 2:
            rgb_indices = [2, 1, 0]
            true_color_composite = image[rgb_indices, ...]
            print(f"Flag {flag} composite is ready")
        else:
            channel_index = flag
            channel = image[channel_index]
            zeros_channel = np.zeros_like(channel)
            true_color_composite = np.stack([channel, zeros_channel, zeros_channel], axis=0)
            print(f"Flag {flag} composite is ready")

        true_color_tensor = torch.from_numpy(true_color_composite).to(dtype=torch.float32)
        true_color_tensor = true_color_tensor.unsqueeze(0)
        low_resolution_input = true_color_tensor.to(DEVICE)
        print("Prepared low resolution input")
        with torch.no_grad():
            high_resolution_output = model(low_resolution_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        print("Hight resolution output is ready")
        if flag == 2:
            bgr_indices = [2, 1, 0]
            scaled_image[:3, ...] = high_resolution_output[bgr_indices, ...]
        else:
            channel_index = flag
            scaled_image[channel_index] = high_resolution_output[0]

        print(f"Flag {flag} finished")

    return scaled_image


def get_metadata(path: str) -> dict:
    with rasterio.open(path) as raster:
        return raster.meta.copy()


def save_raster(image: np.ndarray, meta: dict, output_path: str) -> None:
    num_channels, new_height, new_width = image.shape

    scale_factor = 4

    original_transform = meta['transform']
    new_transform = rasterio.Affine(
        original_transform.a / scale_factor,
        original_transform.b,
        original_transform.c,
        original_transform.d,
        original_transform.e / scale_factor,
        original_transform.f
    )

    meta.update({
        'driver': 'GTiff',
        'count': num_channels,
        'height': new_height,
        'width': new_width,
        'dtype': str(image.dtype),
        'crs': CRS.from_epsg(32639),
        'transform': new_transform,
        'compress': 'lzw'
    })

    with rasterio.open(output_path, 'w', **meta) as output_raster:
        output_raster.write(image)


def main():
    BAND_PATHS = [
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B02_20m.jp2',# Blue
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B03_20m.jp2',# Green
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B04_20m.jp2',# Red
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B05_20m.jp2',# Red Edge 1
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B08_10m.jp2',# NIR
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B8A_20m.jp2',# Red Edge 2
        '../00_src/01_sentinel2/01_raw_data_processed/2017/S2A_MSIL2A_20170728T073611_N0500_R092_T39TTG_20231007T092611.SAFE/T39TTG_20170728T073611_B11_20m.jp2'# SWIR 1
    ]
    OUTPUT_PATH = '../00_src/01_sentinel2/02_tiles/R5m/2017/T39TTG_20170728T073611.tif'
    ESRGAN_MODEL = load_esrgan_model()
    print("Preparation ready")

    bands = [read_raster(band_path) for band_path in BAND_PATHS]
    bands = unify_bands(bands)
    image = np.stack(bands, axis=0)
    print("Bands stacked")

    normalized_image = min_max_normalization(image, band_min_value=0.0, band_max_value=65535.0)  # 16 bit
    print("Bands normalized")

    scaled_image = increase_spatial_resolution(image=normalized_image, model=ESRGAN_MODEL)
    print("Image scaled")

    meta_data = get_metadata(BAND_PATHS[0])
    save_raster(scaled_image, meta=meta_data, output_path=OUTPUT_PATH)


if __name__ == '__main__':
    main()
