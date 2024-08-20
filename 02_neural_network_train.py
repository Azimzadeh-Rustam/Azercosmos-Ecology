import rasterio
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import scienceplots
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate, BatchNormalization,
                          Dropout, Lambda)
from keras import backend as K
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

FONT_SIZE = 14
PATCH_SIZE = 256
NUM_CHANNELS = 4
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


def min_max_normalization(image):
    BAND_MIN_VALUE = 0.0
    BAND_MAX_VALUE = 4095.0

    return (image - BAND_MIN_VALUE) / (BAND_MAX_VALUE - BAND_MIN_VALUE)


def mask_to_label(mask):
    mask_height, mask_width = mask.shape[0], mask.shape[1]

    green_channel = mask[:, :, 1]
    blue_channel = mask[:, :, 2]

    forest_mask = green_channel > 0
    sea_mask = blue_channel > 0

    label = np.zeros((mask_height, mask_width), dtype=np.int32)
    label[forest_mask] = 1
    label[sea_mask] = 2

    return label


def split_into_patches(image):
    patches = list()

    image_height, image_width, num_channels = image.shape

    for start_y in range(0, image_height, PATCH_SIZE):
        for start_x in range(0, image_width, PATCH_SIZE):
            end_y = start_y + PATCH_SIZE
            end_x = start_x + PATCH_SIZE

            patch_height = min(PATCH_SIZE, image_height - start_y)
            patch_width = min(PATCH_SIZE, image_width - start_x)

            patch = np.zeros((PATCH_SIZE, PATCH_SIZE, num_channels), dtype=image.dtype)
            patch[:patch_height, :patch_width, :] = image[start_y:end_y, start_x:end_x, :]

            patches.append(patch)

    return patches


def visualize_patches(image, mask):
    image = image[:, :, :3]
    mask = mask[:, :, :3]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    axis1 = axes[0]
    axis1.imshow(image, vmin=0, vmax=1)
    axis1.axis('off')

    axis2 = axes[1]
    axis2.imshow(mask, vmin=0, vmax=1)
    axis2.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def multi_unet_model(num_classes):

    inputs = Input((PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS))

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


def check_overfitting(history):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    axis1 = axes[0]
    axis1.set_ylabel('Loss Function', fontsize=14)
    axis1.set_xlabel('Number of Epochs', fontsize=14)
    axis1.tick_params(axis='both', labelsize=10)
    axis1.plot(history.history['loss'], label='Train Data')
    axis1.plot(history.history['val_loss'], label='Validation Data')
    axis1.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    axis2 = axes[1]
    axis2.set_ylabel('AUC', fontsize=14)
    axis2.set_xlabel('Number of Epochs', fontsize=14)
    axis2.tick_params(axis='both', labelsize=10)
    axis2.plot(history.history['auc'], label='Train Data')
    axis2.plot(history.history['val_auc'], label='Validation Data')
    axis2.legend(loc='best', fontsize=10, fancybox=False, edgecolor='black')

    plt.show()
    plt.close()


def main():
    image = read_tif('src/img/2017.TIF')
    mask = read_tif('mask.tif')

    image = image[:8960, :, :4]
    mask = mask[:8960, :, :]

    image = min_max_normalization(image)
    mask = min_max_normalization(mask)

    image_patches = split_into_patches(image)
    mask_patches = split_into_patches(mask)

    #for _ in range(20):
    #    random_id = random.randint(0, len(image_patches) - 1)
    #    visualize_patches(image_patches[random_id], mask_patches[random_id])

    label_patches = [mask_to_label(mask_patch) for mask_patch in mask_patches]

    input_train, input_test, output_train, output_test = train_test_split(image_patches, label_patches,
                                                                          test_size=0.3, random_state=42)

    input_train = np.array(input_train)
    input_test = np.array(input_test)
    output_train = np.array(output_train)
    output_test = np.array(output_test)

    total_classes = len(np.unique(label_patches[0]))
    neural_network = multi_unet_model(num_classes=total_classes)

    neural_network.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    training_history = neural_network.fit(input_train, output_train, epochs=150, batch_size=16, verbose=1,
                                          validation_data=(input_test, output_test), shuffle=False)

    neural_network.save('forests_sea_segmentation.h5')

    #check_overfitting(training_history)


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    set_plot_style()
    main()
