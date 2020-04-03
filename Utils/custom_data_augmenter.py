from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment_segmentation_data(images, masks, rotate=False, flip_horizontal=False, flip_vertical=False):
    augmented_images, augmented_masks = images.copy(), masks.copy()
    if rotate:
        num_of_images = augmented_images.shape[0]
        for idx in range(num_of_images):
            # rotate the image by degrees in rotation_angles
            for angle in [90, 270, 360]:
                image_rotated = ImageDataGenerator().apply_transform(X_train[idx], {'theta': angle})
                mask_rotated  = ImageDataGenerator().apply_transform(Y_train[idx], {'theta': angle})
                augmented_images = np.append(augmented_images, image_rotated)
                augmented_masks = np.append(augmented_masks, mask_rotated)
    if flip_horizontal:
        num_of_images = augmented_images.shape[0]
        # flip each image horizontally
        for idx in range(num_of_images):
            image_fliped_horizontally = ImageDataGenerator().apply_transform(X_train[idx], {'flip_horizontal': True})
            mask_fliped_horizontally  = ImageDataGenerator().apply_transform(Y_train[idx], {'flip_horizontal': True})
            augmented_images = np.append(augmented_images, image_fliped_horizontally)
            augmented_masks = np.append(augmented_masks, mask_fliped_horizontally)
    if flip_vertical:
        num_of_images = augmented_images.shape[0]
        # flip each image vertically
        for idx in range(num_of_images):
            image_fliped_vertically = ImageDataGenerator().apply_transform(X_train[idx], {'flip_vertical': True})
            mask_fliped_vertically  = ImageDataGenerator().apply_transform(Y_train[idx], {'flip_vertical': True})
            augmented_images = np.append(augmented_images, image_fliped_vertically)
            augmented_masks = np.append(augmented_masks, mask_fliped_vertically)
    return (augmented_images, augmented_masks)
