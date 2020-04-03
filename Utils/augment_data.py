from keras.preprocessing.image import ImageDataGenerator

def augment_segmentation_data(images, masks, rotate=False, flip_horizontal=False, flip_vertical=False):
    augmented_images, augmented_masks = images.copy(), masks.copy()
    if rotate:
        num_of_images = augmented_images.shape[0]
        for idx in range(num_of_images):
            # rotate the image by degrees in rotation_angles
            for angle in [90, 270, 360]:
                rotated_image = ImageDataGenerator().apply_transform(X_train[idx], {'theta': angle})
                rotated_mask  = ImageDataGenerator().apply_transform(Y_train[idx], {'theta': angle})
                augmented_images.append(rotated_image)
                augmented_masks.append(rotated_mask)
    if flip_horizontal:
        num_of_images = augmented_images.shape[0]
        # flip each image horizontally
        for idx in range(num_of_images):
            rotated_image = ImageDataGenerator().apply_transform(X_train[idx], {'flip_horizontal': True})
            rotated_mask  = ImageDataGenerator().apply_transform(Y_train[idx], {'flip_horizontal': True})
            augmented_images.append(rotated_image)
            augmented_masks.append(rotated_mask)
    if flip_vertical:
        num_of_images = augmented_images.shape[0]
        # flip each image vertically
        for idx in range(num_of_images):
            rotated_image = ImageDataGenerator().apply_transform(X_train[idx], {'flip_vertical': True})
            rotated_mask  = ImageDataGenerator().apply_transform(Y_train[idx], {'flip_vertical': True})
            augmented_images.append(rotated_image)
            augmented_masks.append(rotated_mask)
    return (augmented_images, augmented_masks)
