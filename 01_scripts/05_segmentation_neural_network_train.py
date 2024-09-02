import random
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
from skimage.transform import resize
import scienceplots

FONT_SIZE = 14
PATCH_SIZE = 256
INPUT_CHANNELS = 4
MY_FORMATTER = ScalarFormatter(useMathText=True)
MY_FORMATTER.set_scientific(True)
MY_FORMATTER.set_powerlimits((-1, 1))


def set_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'pdf.fonttype': 42,
        'axes.formatter.limits': (-1, 1),
        'axes.formatter.useoffset': True,
        'axes.formatter.offset_threshold': 1
    })


def read_tif(path):
    with rasterio.open(path) as raster:
        data = raster.read()
        data = data.transpose((1, 2, 0))
        return data.astype(np.float32)


def resample_image(image, current_resolution, target_resolution):
    scale_factor = current_resolution / target_resolution

    initial_height, initial_width = image.shape[0], image.shape[1]

    new_height = int(initial_height * scale_factor)
    new_width = int(initial_width * scale_factor)

    return resize(image, (new_height, new_width), order=0, mode='constant', cval=0, anti_aliasing=True,
                  preserve_range=True)


def min_max_normalization(image, band_min_value, band_max_value):
    return (image - band_min_value) / (band_max_value - band_min_value)


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


def one_hot_encoding(mask):
    mask_height, mask_width = mask.shape[0], mask.shape[1]
    num_classes = 3

    green_channel = mask[:, :, 1]
    blue_channel = mask[:, :, 2]

    forest_mask = green_channel > 0
    sea_mask = blue_channel > 0
    background_mask = ~(forest_mask | sea_mask)

    one_hot_label = np.zeros((mask_height, mask_width, num_classes), dtype=np.uint8)
    one_hot_label[forest_mask, 1] = 1
    one_hot_label[sea_mask, 2] = 2

    return one_hot_label


def jaccard_index(y_true, y_prediction):
    y_true_flatten = bk.flatten(y_true)
    y_prediction_flatten = bk.flatten(y_prediction)
    intersection = bk.sum(y_true_flatten * y_prediction_flatten)
    index_value = (intersection + 1.0) / (bk.sum(y_true_flatten) + bk.sum(y_prediction_flatten) - intersection + 1.0)
    return index_value


def multi_unet_model(patch_height, patch_width, num_input_channels, num_classes):

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


def plot_history(history):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_ylabel('Loss Function', fontsize=14)
    axis1.set_xlabel('Number of Epochs', fontsize=14)
    axis1.tick_params(axis='both', labelsize=10)
    axis1.plot(history.history['loss'], label='Train Data')
    axis1.plot(history.history['val_loss'], label='Validation Data')
    axis1.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_ylabel('Jaccard Index', fontsize=14)
    axis2.set_xlabel('Number of Epochs', fontsize=14)
    axis2.tick_params(axis='both', labelsize=10)
    axis2.plot(history.history['jaccard_index'], label='Train IoU')
    axis2.plot(history.history['val_jaccard_index'], label='Validation IoU')
    axis2.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


def visualize_patches(model, images, masks, number):
    num_images = images.shape[0]
    colors = np.array([
        [0, 0, 0],  # Background
        [0, 1, 0],  # Forests
        [0, 0, 1]  # Sea
    ])

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    for _ in range(number):
        random_id = random.randint(0, num_images)
        image = images[random_id, :, :, :4]
        prediction = model.predict(image)
        prediction = np.argmax(prediction, axis=-1)
        mask = masks[random_id, :, :, :3]

        axis1 = axes[0]
        axis1.imshow(image[..., :3], vmin=0, vmax=1)
        axis1.axis('off')

        axis2 = axes[1]
        axis2.imshow(mask, vmin=0, vmax=1)
        axis2.axis('off')

        axis2 = axes[2]
        prediction_map = colors[prediction]
        axis2.imshow(prediction_map, vmin=0, vmax=1)
        axis2.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()


def main():
    image = read_tif('../01_src/03_images/2017.TIF')
    mask = read_tif('../01_src/mask.tif')

    image = image[:8960, :, :4]
    mask = mask[:8960, :, :]

    #image = resample_image(image, current_resolution=1.5, target_resolution=30.0)
    #mask = resample_image(mask, current_resolution=1.5, target_resolution=30.0)

    image = min_max_normalization(image, band_min_value=0.0, band_max_value=4095.0)
    mask = min_max_normalization(mask, band_min_value=0.0, band_max_value=255.0)

    multiple_image = pad_to_multiple(image, patch_size=PATCH_SIZE)
    multiple_mask = pad_to_multiple(mask, patch_size=PATCH_SIZE)

    image_patches = split_into_patches(multiple_image, patch_size=PATCH_SIZE)
    mask_patches = split_into_patches(multiple_mask, patch_size=PATCH_SIZE)
    label_patches = [one_hot_encoding(mask_patch) for mask_patch in mask_patches]

    image_patches = np.array(image_patches)
    mask_patches = np.array(mask_patches)
    label_patches = np.array(label_patches)

    total_classes = len(np.unique(label_patches))

    input_train, input_test, output_train, output_test = train_test_split(image_patches, label_patches,
                                                                          test_size=0.3, random_state=42)

    neural_network = multi_unet_model(patch_height=PATCH_SIZE, patch_width=PATCH_SIZE,
                                      num_input_channels=INPUT_CHANNELS, num_classes=total_classes)
    neural_network.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy", jaccard_index])
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)]
    training_history = neural_network.fit(input_train, output_train, epochs=150, batch_size=16, verbose=1,
                                          callbacks=callbacks, validation_split=0.2, shuffle=True)
    neural_network.save('../03_results/01_neural_network/forests_sea_segmentation.h5')

    plot_history(training_history)
    #plot_roc_curve(neural_network, input_test, output_test)
    visualize_patches(neural_network, input_test, output_test, 20)


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    set_plot_style()
    main()
