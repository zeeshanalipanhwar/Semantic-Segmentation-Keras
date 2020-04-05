import numpy as np

def split_image_into_subimages(image, sub_image_size, overlap_ratio):
    W_overlap = round(sub_image_size[0] * overlap_ratio)
    H_overlap = round(sub_image_size[1] * overlap_ratio)
    subimages = []
    for i in range(0, image.shape[0], sub_image_size[0]-W_overlap):
        x1 = i
        if x1+sub_image_size[0] <= image.shape[0]: x2 = x1+sub_image_size[0]
        else: x1, x2 = image.shape[0]-sub_image_size[0], image.shape[0]
        for j in range(0, image.shape[1], sub_image_size[1]-H_overlap):
            y1 = j
            if y1+sub_image_size[1] <= image.shape[1]: y2 = y1+sub_image_size[1]
            else: y1, y2 = image.shape[1]-sub_image_size[1], image.shape[1]
            subimages.append(image[x1:x2, y1:y2, :].copy())
    subimages = np.array(subimages)
    return subimages

def merge_subimages_into_image(subimages, image_size, overlap_ratio):
    W_overlap = round(subimages.shape[0] * overlap_ratio)
    H_overlap = round(subimages.shape[1] * overlap_ratio)

    count = 0
    image = np.zeros((image_size[0], image_size[1], subimages[0].shape[-1]))
    for i in range(0, image_size[0], W_overlap):
        if x1+subimages.shape[0] <= image_size[0]: x2 = x1+subimages.shape[0]
        else: x1, x2 = image_size[0]-subimages.shape[0], image_size[0]
        for j in range(0, image_size[1], H_overlap):
            if y1+subimages.shape[1] <= image_size[1]: y2 = y1+subimages.shape[1]
            else: y1, y2 = image_size[1]-subimages.shape[1], image_size[1]
            image[x1:x2, y1:y2, :] = subimages[count].copy()
            count += 1
    return image
