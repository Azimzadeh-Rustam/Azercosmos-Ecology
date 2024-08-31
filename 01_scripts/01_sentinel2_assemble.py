import matplotlib.pyplot as plt
import rasterio
import numpy as np
import torch
import RRDBNet_arch as arch

PATCH_SIZE = 256
DEVICE = torch.device('cuda')


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as raster:
        array = raster.read()
        return array.astype(np.float32)


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image - band_min_value) / (band_max_value - band_min_value)


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

    return np.pad(image, paddings, mode='constant', constant_values=0)


def increase_spatial_resolution(image: np.ndarray, model: torch.nn.Module, patch_size: int) -> np.ndarray:
    true_color_composite = image[[2, 1, 0], ...]
    num_channels, initial_height, initial_width = true_color_composite.shape

    padded_image = pad_to_multiple(true_color_composite, patch_size)
    _, padded_height, padded_width = padded_image.shape

    scale_factor = 4

    scaled_padded_height = padded_height * scale_factor
    scaled_padded_width = padded_width * scale_factor
    scaled_padded_image = np.zeros((num_channels, scaled_padded_height, scaled_padded_width))

    for start_y in range(0, padded_height, patch_size):
        for start_x in range(0, padded_width, patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size
            patch = padded_image[:, start_y:end_y, start_x:end_x]
            # print(f'Initial patch: {patch.shape}')

            patch_input = torch.from_numpy(patch).float()
            patch_input = patch_input.unsqueeze(0)
            patch_input = patch_input.to(DEVICE)
            with torch.no_grad():
                scaled_patch = model(patch_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            # print(f'Scaled patch: {scaled_patch.shape}')
            scaled_start_y = start_y * scale_factor
            scaled_start_x = start_x * scale_factor
            scaled_end_y = scaled_start_y + patch_size * scale_factor
            scaled_end_x = scaled_start_x + patch_size * scale_factor
            scaled_padded_image[:, scaled_start_y:scaled_end_y, scaled_start_x:scaled_end_x] = scaled_patch
            # print('One more processed')

    reconstructed_height = initial_height * scale_factor
    reconstructed_width = initial_width * scale_factor

    return scaled_padded_image[[0, 1, 2], :reconstructed_height, :reconstructed_width]


def main():
    band_paths = [
        '../00_src/02_Sentinel-2/2022/S2B_MSIL2A_20220816T073619_N0400_R092_T39TUG_20220816T091945.SAFE/GRANULE/L2A_T39TUG_A028432_20220816T074200/IMG_DATA/R20m/T39TUG_20220816T073619_B02_20m.jp2', # blue
        '../00_src/02_Sentinel-2/2022/S2B_MSIL2A_20220816T073619_N0400_R092_T39TUG_20220816T091945.SAFE/GRANULE/L2A_T39TUG_A028432_20220816T074200/IMG_DATA/R20m/T39TUG_20220816T073619_B03_20m.jp2', # green
        '../00_src/02_Sentinel-2/2022/S2B_MSIL2A_20220816T073619_N0400_R092_T39TUG_20220816T091945.SAFE/GRANULE/L2A_T39TUG_A028432_20220816T074200/IMG_DATA/R20m/T39TUG_20220816T073619_B04_20m.jp2', # red
        '../00_src/02_Sentinel-2/2022/S2B_MSIL2A_20220816T073619_N0400_R092_T39TUG_20220816T091945.SAFE/GRANULE/L2A_T39TUG_A028432_20220816T074200/IMG_DATA/R20m/T39TUG_20220816T073619_B05_20m.jp2', # red edge
        '../00_src/02_Sentinel-2/2022/S2B_MSIL2A_20220816T073619_N0400_R092_T39TUG_20220816T091945.SAFE/GRANULE/L2A_T39TUG_A028432_20220816T074200/IMG_DATA/R20m/T39TUG_20220816T073619_B8A_20m.jp2', # nir
        '../00_src/02_Sentinel-2/2022/S2B_MSIL2A_20220816T073619_N0400_R092_T39TUG_20220816T091945.SAFE/GRANULE/L2A_T39TUG_A028432_20220816T074200/IMG_DATA/R20m/T39TUG_20220816T073619_B11_20m.jp2' # swir
    ]

    bands = [read_raster(band_path)[0, ...] for band_path in band_paths]
    image = np.stack(bands, axis=0)
    image = image[:3, ...]
    image = min_max_normalization(image, band_min_value=0.0, band_max_value=65535.0)
    print(image.shape)

    esrgan_model = load_esrgan_model()
    image = increase_spatial_resolution(image=image, model=esrgan_model, patch_size=PATCH_SIZE)
    print(image.shape)


if __name__ == '__main__':
    main()
