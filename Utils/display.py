import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def display_image(image, title=None, figsize=(10, 10)): # Show image
    plt.figure(figsize = figsize)
    plt.imshow(image)
    plt.axis('off')
    if title: plt.title(title)
    plt.show()
    return

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
    return

def plot_train_valid_loss(H, EPOCHS, save=False, save_to=None, model_name=None):
    # plot the training and validation losses
    N = np.arange(0, EPOCHS)
    plt.figure()
    plt.plot(N, H.history["loss"], label="Training")
    plt.plot(N, H.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save: plt.savefig("{}{}_loss_plot.png".format(save_to, model_name))
    plt.show()
    return

def plot_train_valid_accuracy(H, EPOCHS, save=False, save_to=None, model_name=None):
    # plot the training and validation accuracies
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["acc"], label="Training")
    plt.plot(N, H.history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if save: plt.savefig("{}{}_accuracy_plot.png".format(save_to, model_name))
    plt.show()
    return

def plot_train_valid_f1_score(H, EPOCHS, save=False, save_to=None, model_name=None):
    # plot the training and validation f1 scores
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["f1_score"], label="Training")
    plt.plot(N, H.history["val_f1_score"], label="Validation")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    if save: plt.savefig("{}{}_f1_score_plot.png".format(save_to, model_name))
    plt.show()
    return
    
def plot_train_valid_performance(H, EPOCHS, save_figures=False, save_to=None, model_name=None):
    if save_figures:
        if not save_to: raise ValueError("'save_to' must not be 'None' when 'save_figures' is 'True'!")
        elif not model_name: raise ValueError("'model_name' must not be 'None' when 'save_figures' is 'True'!")
        else:
            plot_train_valid_loss(H, EPOCHS, save_figures, save_to, model_name)
            plot_train_valid_accuracy(H, EPOCHS, save_figures, save_to, model_name)
            plot_train_valid_f1_score(H, EPOCHS, save_figures, save_to, model_name)
    else: # save_figures is False
        plot_train_valid_loss(H, EPOCHS, save_figures, save_to, model_name)
        plot_train_valid_accuracy(H, EPOCHS, save_figures, save_to, model_name)
        plot_train_valid_f1_score(H, EPOCHS, save_figures, save_to, model_name)
    return
