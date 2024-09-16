import os
import numpy as np
import rasterio
from keras.src.layers import Activation, UpSampling2D
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout,
                                     BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as bk
from skimage.transform import resize
import scienceplots

FONT_SIZE = 14
PATCH_SIZE = 256
INPUT_CHANNELS = 4
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-2, 2))


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-2, 2),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def read_raster(path):
    with rasterio.open(path) as raster:
        return raster.read()


def get_all_tiff(folder_path: str) -> np.ndarray:
    images = list()
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            image = read_raster(file_path)
            images.append(image)

    return np.stack(images, axis=0)


def pad_to_multiple(image, patch_size):
    height, width = image.shape[0], image.shape[1]

    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    if pad_height == 0 and pad_width == 0:
        return image

    paddings = [(0, pad_height), (0, pad_width), (0, 0)]

    return np.pad(image, paddings, mode='constant', constant_values=0)


def split_into_patches(image, patch_size):
    patches = list()

    image_height, image_width, num_channels = image.shape

    for start_y in range(0, image_height, patch_size):
        for start_x in range(0, image_width, patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size

            patch = image[start_y:end_y, start_x:end_x, :]
            patches.append(patch)

    return patches


def multi_unet_model(patch_height, patch_width, num_input_channels):
    inputs = Input((patch_height, patch_width, num_input_channels))

    # Encoder
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Decoder
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output layer
    outputs = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

    return Model(inputs=inputs, outputs=outputs)


def plot_history(history):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_ylabel('Loss Function', fontsize=FONT_SIZE)
    axis1.set_xlabel('Number of Epochs', fontsize=FONT_SIZE)
    axis1.tick_params(axis='both', labelsize=FONT_SIZE)
    axis1.plot(history.history['loss'], label='Train Data')
    axis1.plot(history.history['val_loss'], label='Validation Data')
    axis1.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_ylabel('Accuracy', fontsize=FONT_SIZE)
    axis2.set_xlabel('Number of Epochs', fontsize=FONT_SIZE)
    axis2.tick_params(axis='both', labelsize=FONT_SIZE)
    axis2.plot(history.history['accuracy'], label='Train IoU')
    axis2.plot(history.history['val_accuracy'], label='Validation IoU')
    axis2.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


def main():
    FILES_FOLDER_PATH = '../00_src/01_sentinel2/03_full_area'

    images = get_all_tiff(FILES_FOLDER_PATH)

    all_image_patches = list()
    all_label_patches = list()
    for image in images:
        multiple_image = pad_to_multiple(image, patch_size=PATCH_SIZE)

        b_g_r_nir_channels_order = [0, 1, 2, 4]
        input_image = multiple_image[..., b_g_r_nir_channels_order]
        input_image = np.nan_to_num(input_image, nan=-1)

        rd1_rd4_swir_channels_order = [3, 5, 6]
        label = multiple_image[..., rd1_rd4_swir_channels_order]

        image_patches = split_into_patches(input_image, patch_size=PATCH_SIZE)
        label_patches = split_into_patches(label, patch_size=PATCH_SIZE)

        all_image_patches.append(image_patches)
        all_label_patches.append(label_patches)
    all_image_patches = np.concatenate(all_image_patches, axis=0)
    all_label_patches = np.concatenate(all_label_patches, axis=0)

    input_train, input_test, output_train, output_test = train_test_split(all_image_patches, all_label_patches,
                                                                          test_size=0.3, random_state=42)

    neural_network = multi_unet_model(patch_height=PATCH_SIZE, patch_width=PATCH_SIZE,
                                      num_input_channels=INPUT_CHANNELS)
    neural_network.compile(optimizer="adam", loss='mean_squared_error', metrics=["accuracy"])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)]
    training_history = neural_network.fit(input_train, output_train, epochs=150, batch_size=16, verbose=1,
                                          callbacks=callbacks, validation_split=0.2, shuffle=True)
    neural_network.save('../02_results/01_neural_networks/01_spectra_reconstruction/01_pre-trained_model/spectra_reconstruction_R20m.keras')

    plot_history(training_history)


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    set_plot_style()
    main()
