import os
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as bk
import scienceplots

FONT_SIZE = 14
PATCH_SIZE = 256
NUM_INPUT_CHANNELS = 3
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-2, 2))

colors = np.array([
        #R  G  B
        [0, 0, 0],   # Background - Black
        [0, 255, 0], # Forests - Green
        [0, 0, 255]  # Sea - Blue
    ])


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-2, 2),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as raster:
        array = raster.read()
        return array.transpose((1, 2, 0)) # (height, width, num_channels)


def get_all_tiff(folder_path: str) -> np.ndarray:
    images = list()
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            file_path = os.path.join(folder_path, filename)
            image = read_raster(file_path)
            images.append(image)

    return np.stack(images, axis=0)


def pad_to_multiple(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape[0], image.shape[1]

    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    paddings = [(0, pad_height), (0, pad_width), (0, 0)]

    return np.pad(image, paddings, mode='constant', constant_values=np.nan)


def split_into_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    patches = list()

    image_height, image_width, _ = image.shape

    for start_y in range(0, image_height, patch_size):
        for start_x in range(0, image_width, patch_size):
            end_y = start_y + patch_size
            end_x = start_x + patch_size

            patch = image[start_y:end_y, start_x:end_x, :]
            patches.append(patch)

    return np.array(patches)


def one_hot_encoding(image: np.ndarray) -> np.ndarray:
    NDVI_THRESHOLD_LOW = 0.36
    NDVI_THRESHOLD_HIGH = 1.0

    MNDWI_THRESHOLD_LOW = 0.0
    MNDWI_THRESHOLD_HIGH = 1.0

    mask_height, mask_width, _ = image.shape
    num_classes = 3

    green_channel = image[..., 1]
    red_channel = image[..., 2]
    nir_channel = image[..., 4]
    swir1_channel = image[..., 6]

    ndvi_channel = (nir_channel - red_channel) / (nir_channel + red_channel + 1e-10)
    forest_mask = np.where((ndvi_channel > NDVI_THRESHOLD_LOW) & (ndvi_channel < NDVI_THRESHOLD_HIGH), True, False)

    mndwi_channel = (green_channel - swir1_channel) / (green_channel + swir1_channel + 1e-10)
    sea_mask = np.where((mndwi_channel > MNDWI_THRESHOLD_LOW) & (mndwi_channel < MNDWI_THRESHOLD_HIGH), True, False)

    background_mask = ~(forest_mask | sea_mask)

    one_hot_label = np.zeros((mask_height, mask_width, num_classes), dtype=np.uint8)
    one_hot_label[background_mask, 0] = 1
    one_hot_label[forest_mask, 1] = 1
    one_hot_label[sea_mask, 2] = 1

    return one_hot_label


def calculate_class_weights(mask: np.array) -> dict:
    class_pixel_counts = np.bincount(mask.flatten())
    num_classes = len(class_pixel_counts)
    total_pixels = mask.size
    return {i: total_pixels / (num_classes * np.maximum(class_pixel_counts[i], 1)) for i in range(num_classes)}


def jaccard_index(y_true: tf.Tensor, y_prediction: tf.Tensor) -> tf.Tensor:
    y_true_flatten = bk.flatten(y_true)
    y_prediction_flatten = bk.flatten(y_prediction)
    intersection = bk.sum(y_true_flatten * y_prediction_flatten)
    index_value = (intersection + 1.0) / (bk.sum(y_true_flatten) + bk.sum(y_prediction_flatten) - intersection + 1.0)
    return index_value


def multi_unet_model(patch_height: int, patch_width: int, num_input_channels: int, num_classes: int) -> Model:

    inputs = Input((patch_height, patch_width, num_input_channels))

    source_input = inputs

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(source_input)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def min_max_normalization(image: np.ndarray, band_min_value: float, band_max_value: float) -> np.ndarray:
    return (image - band_min_value) / (band_max_value - band_min_value)


def normalize_with_clipping(image: np.ndarray) -> np.ndarray:
    p_low, p_high = np.nanpercentile(image, [0.5, 99.5])
    clipped_image = np.clip(image, p_low, p_high)
    return min_max_normalization(clipped_image, p_low, p_high)


def full_predict(model: Model, image: np.ndarray) -> np.ndarray:
    initial_height, initial_width, _ = image.shape

    multiple_image = pad_to_multiple(image=image, patch_size=PATCH_SIZE)
    multiple_height, multiple_width, _ = multiple_image.shape

    input_image = multiple_image[..., :3]
    input_image = np.nan_to_num(input_image, nan=-1)

    prediction = np.zeros((multiple_height, multiple_width), dtype=np.uint8)

    for start_y in range(0, multiple_height, PATCH_SIZE):
        for start_x in range(0, multiple_width, PATCH_SIZE):
            end_y = start_y + PATCH_SIZE
            end_x = start_x + PATCH_SIZE

            input_patch = input_image[start_y:end_y, start_x:end_x, :]
            predicted_patch_label = model.predict(np.expand_dims(input_patch, axis=0))
            predicted_patch = np.argmax(predicted_patch_label.squeeze(), axis=-1)

            prediction[start_y:end_y, start_x:end_x] = predicted_patch

    return prediction[:initial_height, :initial_width]


def compare_full_prediction(model: Model, image: np.ndarray) -> None:
    rgb_indices = [2, 1, 0]
    true_color_composite = image[..., rgb_indices]
    normalized_image = normalize_with_clipping(true_color_composite)

    label = one_hot_encoding(image)
    mask = np.argmax(label, axis=-1)
    mask_map = colors[mask]

    prediction = full_predict(model=model, image=image)
    prediction_map = colors[prediction]

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    axis1 = axes[0]
    axis1.imshow(normalized_image, vmin=0, vmax=1)
    axis1.set_title("Satellite Image")
    axis1.axis('off')

    axis2 = axes[1]
    axis2.imshow(mask_map, vmin=0, vmax=255)
    axis2.set_title("True Mask")
    axis2.axis('off')

    axis3 = axes[2]
    axis3.imshow(prediction_map, vmin=0, vmax=255)
    axis3.set_title("NN Prediction")
    axis3.axis('off')

    plt.savefig(f'../02_results/03_neural_networks/02_forests_sea_segmentation/02_performance_plots/full_prediction.png', dpi=300)
    plt.close()


def compare_patches_predictions(model: Model, images: np.ndarray, labels: np.ndarray, number: int) -> None:
    num_images = images.shape[0]

    for i in range(number):
        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

        random_id = np.random.randint(0, num_images - 1)

        image = images[random_id, ...]
        rgb_indices = [2, 1, 0]
        true_color_composite = image[..., rgb_indices]
        normalized_image = normalize_with_clipping(true_color_composite)
        axis1 = axes[0]
        axis1.imshow(normalized_image, vmin=0, vmax=1)
        axis1.set_title("Satellite Image")
        axis1.axis('off')

        label = labels[random_id]
        mask = np.argmax(label, axis=-1)
        mask_map = colors[mask]
        axis2 = axes[1]
        axis2.imshow(mask_map, vmin=0, vmax=255)
        axis2.set_title("True Mask")
        axis2.axis('off')

        prediction_label = model.predict(np.expand_dims(image, axis=0))
        prediction = np.argmax(prediction_label.squeeze(), axis=-1)
        prediction_map = colors[prediction]
        axis3 = axes[2]
        axis3.imshow(prediction_map, vmin=0, vmax=255)
        axis3.set_title("NN Prediction")
        axis3.axis('off')

        plt.savefig(f'../02_results/03_neural_networks/02_forests_sea_segmentation/02_performance_plots/compare_{i}', dpi=300)
        plt.close()


def plot_history(history) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_ylabel('Categorical Crossentropy', fontsize=FONT_SIZE)
    axis1.set_xlabel('Number of Epochs', fontsize=FONT_SIZE)
    axis1.xaxis.set_major_formatter(MY_FORMATTER)
    axis1.yaxis.set_major_formatter(MY_FORMATTER)
    axis1.xaxis.get_offset_text().set_size(FONT_SIZE)
    axis1.yaxis.get_offset_text().set_size(FONT_SIZE)
    axis1.tick_params(axis='both', labelsize=FONT_SIZE)
    axis1.plot(history.history['loss'], label='Train Data')
    axis1.plot(history.history['val_loss'], label='Validation Data')
    axis1.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_ylabel('Jaccard Index', fontsize=FONT_SIZE)
    axis2.set_xlabel('Number of Epochs', fontsize=FONT_SIZE)
    axis2.xaxis.set_major_formatter(MY_FORMATTER)
    axis2.yaxis.set_major_formatter(MY_FORMATTER)
    axis2.xaxis.get_offset_text().set_size(FONT_SIZE)
    axis2.yaxis.get_offset_text().set_size(FONT_SIZE)
    axis2.tick_params(axis='both', labelsize=FONT_SIZE)
    axis2.plot(history.history['jaccard_index'], label='Train IoU')
    axis2.plot(history.history['val_jaccard_index'], label='Validation IoU')
    axis2.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')

    plt.savefig(f'../02_results/03_neural_networks/02_forests_sea_segmentation/02_performance_plots/training_history.png', dpi=300)
    plt.close()


def main() -> None:
    FILES_FOLDER_PATH = '../00_src/01_sentinel2/03_full_area'
    AREA_OF_INTEREST_PATH = '../00_src/01_sentinel2/04_aoi/20230821T073619.tif'

    images = get_all_tiff(FILES_FOLDER_PATH)

    all_image_patches = list()
    all_label_patches = list()
    for image in images:
        multiple_image = pad_to_multiple(image, patch_size=PATCH_SIZE)
        label = one_hot_encoding(multiple_image)

        input_image = multiple_image[..., :3]
        input_image = np.nan_to_num(input_image, nan=-1)

        image_patches = split_into_patches(input_image, patch_size=PATCH_SIZE)
        label_patches = split_into_patches(label, patch_size=PATCH_SIZE)

        all_image_patches.append(image_patches)
        all_label_patches.append(label_patches)
    all_image_patches = np.concatenate(all_image_patches, axis=0)
    all_label_patches = np.concatenate(all_label_patches, axis=0)

    image_2023 = images[6]
    label_2023 = one_hot_encoding(image_2023)
    mask_2023 = np.argmax(label_2023, axis=-1)
    class_weights = calculate_class_weights(mask_2023)
    total_classes = 3

    input_train, input_test, output_train, output_test = train_test_split(all_image_patches, all_label_patches,
                                                                          test_size=0.3, shuffle=True)

    neural_network = multi_unet_model(patch_height=PATCH_SIZE, patch_width=PATCH_SIZE,
                                      num_input_channels=NUM_INPUT_CHANNELS, num_classes=total_classes)
    neural_network.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy", jaccard_index])
    callbacks = [EarlyStopping(monitor='val_jaccard_index', mode='max', patience=20, restore_best_weights=True, verbose=1)]
    training_history = neural_network.fit(input_train, output_train, class_weight=class_weights, epochs=100,
                                          batch_size=32, verbose=1, callbacks=callbacks, validation_split=0.2,
                                          shuffle=True)
    neural_network.save('../02_results/03_neural_networks/02_forests_sea_segmentation/01_pre-trained_model/forests_sea_segmentation_R20m.keras')

    area_of_interest = read_raster(AREA_OF_INTEREST_PATH)
    compare_full_prediction(model=neural_network, image=area_of_interest)
    compare_patches_predictions(model=neural_network, images=input_test, labels=output_test, number=50)
    plot_history(training_history)


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    set_plot_style()
    main()
