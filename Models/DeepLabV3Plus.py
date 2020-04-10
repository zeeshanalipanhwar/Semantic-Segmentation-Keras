from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Input, concatenate, UpSampling2D
from keras import backend as K

class DeepLabV3Plus:
    def __init__(self, depth):
        self.depth = depth

    def atrous_spatial_pyramid_pooling(self, input_layer, depth, dropout):
        conv11_layer = Conv2D(depth, (1, 1), activation='relu', padding="same")(input_layer)
        conv11_layer = BatchNormalization()(conv11_layer)
        conv11_layer = Dropout(dropout)(conv11_layer)
        
        atrous_conv1 = Conv2D(depth, (3, 3), dilation_rate=(6 , 6 ), activation='relu', padding="same")(input_layer)
        atrous_conv1 = BatchNormalization()(atrous_conv1)
        atrous_conv1 = Dropout(dropout)(atrous_conv1)

        atrous_conv2 = Conv2D(depth, (3, 3), dilation_rate=(12, 12), activation='relu', padding="same")(input_layer)
        atrous_conv2 = BatchNormalization()(atrous_conv2)
        atrous_conv2 = Dropout(dropout)(atrous_conv2)

        atrous_conv3 = Conv2D(depth, (3, 3), dilation_rate=(18, 18), activation='relu', padding="same")(input_layer)
        atrous_conv3 = BatchNormalization()(atrous_conv3)
        atrous_conv3 = Dropout(dropout)(atrous_conv3)

        return conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3

    def encoder(self, input_layer, dropout = 0.25):
        # Block one that reduces the input shape by 2
        output_layer = Conv2D(self.depth, (3, 3), activation='relu', padding="same")(input_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Conv2D(self.depth, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)

        output_forwd = output_layer
        
        # Block two that reduces the input shape by 4
        output_layer = Conv2D(self.depth*2, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Conv2D(self.depth*2, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block three that reduces the input shape by 8
        output_layer = Conv2D(self.depth*4, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Conv2D(self.depth*4, (3, 3), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = MaxPooling2D(pool_size=(2, 2))(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block four that retains 8 using Atrous Convolution
        output_layer = Conv2D(self.depth*8, (3, 3), dilation_rate=(2,2), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Conv2D(self.depth*8, (3, 3), dilation_rate=(2,2), activation='relu', padding="same")(output_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Dropout(dropout)(output_layer)
        
        # Block five of atrous spatial pyramid pooling and image max pooling
        conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3 = self.atrous_spatial_pyramid_pooling(output_layer, self.depth*8, dropout)
        maxpooled_in = MaxPooling2D(pool_size=(8, 8))(input_layer)

        # Block six of concatination and then a 1x1 conv on the concatenated output
        concatenated = concatenate([conv11_layer, atrous_conv1, atrous_conv2, atrous_conv3, maxpooled_in])        
        output_layer = Conv2D(1, (1, 1), activation="relu", padding="same")(concatenated)
        
        return output_forwd, output_layer
    
    def equal(a, b):
        '''
        Inputs:
            a, b: n, m dimentional tensors where n, m >= 1
        Returns:
            True: when the tensors a and b are of the same shape
            False: otherwise
        '''
        if len(a.shape) != len(b.shape): return False
        if ((a.shape[1:] == b.shape[1:]): return True
        for i in len(a.shape):
            if not a.shape[i] and not b.shape[i]: pass
            else: return False
        return True
            
    def decoder(self, output_forwd, output_layer):
        # Block one of 4 times upsampling of the output using Bilinear interpolation
        output_layer_upsampled = UpSampling2D(size=(4, 4), data_format=None, interpolation='bilinear')(output_layer)
    
        # Block two of reducing depth of the output of the fourth block to one to equate it to that of the output of encoder block six
        output_forwd_d_reduced = Conv2D(1, (1, 1), activation="relu", padding="same")(output_forwd)
        
        # Make sure the shape of both is the same
        if self.equal(output_forwd_d_reduced, output_layer_upsampled): pass
        else: raise ValueError("Shapes of 'output_forwd_d_reduced' and 'output_layer_upsampled' are expected to be same, but got {} and {}!".format(output_forwd_d_reduced.shape, output_layer_upsampled.shape))

        concatenated = concatenate([output_forwd_d_reduced, output_layer_upsampled])
        decoded_out = Conv2D(1, (3, 3), activation="relu", padding="same")(concatenated)
        
        # Block three of 2 times upsampling of the above decoded output using Bilinear interpolation
        decoded_out = UpSampling2D(size=(2, 2), data_format=None, interpolation='bilinear')(decoded_out)
        
        return decoded_out
        
    def DeepLabV3Plus(self, input_shape, dropout = 0.25):
        input_layer = Input(shape=input_shape)        

        output_forwd, output_layer = self.encoder(input_layer, dropout)
        output_layer = self.decoder(output_forwd, output_layer)
        output_layer = Activation("sigmoid")(output_layer)
        
        # Create the model using the input layer and the final output layer
        model = Model(input_layer, output_layer)
        return model
