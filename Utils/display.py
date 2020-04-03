import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def display(image, title=None): # Show image
    plt.figure(figsize = (10, 10))
    plt.imshow(image)
    plt.axis('off')
    if title: plt.title(title)
    plt.show()

def display_masked(image, mask, image_name="image", mask_name="mask", cells_color=[1, 1, 0], figsize = (20, 20)):
    '''
    Show image with its segmentation mask
    '''
    # resize mask as a three channel image for ploting
    mask_resized = image.copy()
    mask_resized[:,:,0] = mask[:,:,0]
    mask_resized[:,:,1] = mask[:,:,0]
    mask_resized[:,:,2] = mask[:,:,0]

    # create a masked image
    mask_ = mask.copy().round().astype(int)
    masked_image = image.copy()
    masked_image[:,:,0][mask_[:,:,0]==1] = cells_color[0]
    masked_image[:,:,1][mask_[:,:,0]==1] = cells_color[1]
    masked_image[:,:,2][mask_[:,:,0]==1] = cells_color[2]

    plt.figure(figsize = figsize)
    plt.subplot(1,3,1)
    plt.imshow(image, 'gray')
    plt.title(image_name)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(mask_resized, 'gray')
    plt.title(mask_name)
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(masked_image, 'gray')
    plt.title("{} with {} overlapped".format(image_name, mask_name))
    plt.axis('off')
    plt.show()
