from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def rotate_segmentation_data(images, masks, percent):
    num_of_images = images.shape[0]
    # include the origional instances to the final list of augmented data
    images_rotated, masks_rotated = list(images), list(masks)
    for idx in range(0, num_of_images, int(1/percent)): # 1/percent is the step size
        # rotate the image and its mask by degrees in [90, 270, 360]
        for angle in [90, 270, 360]:
            image_rotated = ImageDataGenerator().apply_transform(images[idx], {'theta': angle})
            mask_rotated  = ImageDataGenerator().apply_transform(masks[idx], {'theta': angle})
            images_rotated.append(image_rotated)
            masks_rotated.append(mask_rotated)
    images_rotated = np.array(images_rotated)
    masks_rotated = np.array(masks_rotated)
    return images_rotated, masks_rotated

def fliped_segmentation_data_horizontally(images, masks, percent):
    num_of_images = images.shape[0]
    # include the origional instances to the final list of augmented data
    images_fliped_horizontally, masks_fliped_horizontally = list(images), list(masks)
    for idx in range(0, num_of_images, int(1/percent)): # 1/percent is the step size
        # flip the image and its mask horizontally
        image_fliped_horizontally = ImageDataGenerator().apply_transform(images[idx], {'flip_horizontal': True})
        mask_fliped_horizontally  = ImageDataGenerator().apply_transform(masks[idx], {'flip_horizontal': True})
        images_fliped_horizontally.append(image_fliped_horizontally)
        masks_fliped_horizontally.append(mask_fliped_horizontally)
    images_fliped_horizontally = np.array(images_fliped_horizontally)
    masks_fliped_horizontally = np.array(masks_fliped_horizontally)
    return images_fliped_horizontally, masks_fliped_horizontally

def fliped_segmentation_data_vertically(images, masks, percent):
    num_of_images = images.shape[0]
    # include the origional instances to the final list of augmented data
    images_fliped_vertically, masks_fliped_vertically = list(images), list(masks)
    for idx in range(0, num_of_images, int(1/percent)): # 1/percent is the step size
        # flip the image and its mask vertically
        image_fliped_vertically = ImageDataGenerator().apply_transform(images[idx], {'flip_vertical': True})
        mask_fliped_vertically  = ImageDataGenerator().apply_transform(masks[idx], {'flip_vertical': True})
        images_fliped_vertically.append(image_fliped_vertically)
        masks_fliped_vertically.append(mask_fliped_vertically)
    images_fliped_vertically = np.array(images_fliped_vertically)
    masks_fliped_vertically = np.array(masks_fliped_vertically)
    return images_fliped_vertically, masks_fliped_vertically

def augment_segmentation_data(images, masks, rotate=False, flip_horizontal=False, flip_vertical=False,
                              rotate_percent=1, flip_horizontal_percent=1, flip_vertical_percent=1):
    augmented_images, augmented_masks = images.copy(), masks.copy()
    if rotate:
        augmented_images, augmented_masks = rotate_segmentation_data(augmented_images, augmented_masks, rotate_percent)
    if flip_horizontal:
        augmented_images, augmented_masks = fliped_segmentation_data_horizontally(augmented_images, augmented_masks, flip_horizontal_percent)
    if flip_vertical:
        augmented_images, augmented_masks = fliped_segmentation_data_horizontally(augmented_images, augmented_masks, flip_vertical_percent)
    return augmented_images, augmented_masks
