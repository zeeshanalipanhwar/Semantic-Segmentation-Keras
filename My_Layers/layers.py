from keras.layers import Input, Layer
from keras.models import Model
from keras.layers.convolutional import Conv2D
import keras.backend as K

class MaxPool2DWithIndices(Layer):
    def __init__(self, filter_size=(2,2), strides=(2,2), padding='valid', **kwargs):
        super(MaxPool2DWithIndices, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding

    def __call__(self, inputs):
        '''
        inputs: (n, H, W, d), ie batch of n inputs of size (H, W, d)
        '''
        print (inputs.shape)
        inputs_shape = inputs.shape #K.array(inputs.shape)
        pooled_H = (inputs_shape[1]-self.filter_size[0])//self.strides[0]+1
        pooled_W = (inputs_shape[2]-self.filter_size[1])//self.strides[1]+1
        #if type(inputs_shape[0]) is not int: inputs_shape[0] = 1
        mpooled = np.zeros((inputs_shape[0], pooled_H, pooled_W, inputs_shape[3]))
        indices = np.zeros((inputs_shape[0], pooled_H, pooled_W, inputs_shape[3]))
        for n in range(0, inputs_shape[0], 2): # for each example
            for i in range(0, inputs_shape[1], 2):
                for j in range(0, inputs_shape[2], 2):
                    for k in range(inputs_shape[3]):
                        mpooled[n, i//2, j//2, k] = np.max(inputs[n, i:i+2, j:j+2, k])
                        indices[n, i//2, j//2, k] = np.argmax(inputs[n, i:i+2, j:j+2, k])
        return [mpooled, indices]

class MaxUnPool2DWithIndices(Layer):
    def __init__(self, filter_size=(2,2), strides=(2,2), padding='valid'):
        super(MaxUnPool2DWithIndices, self).__init__(**kwargs)
        self.indices = indices
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding

    def __call__(self, inputs, indices):
        '''
        inputs: (n, H, W, d), ie batch of n inputs of size (H, W, d)
        '''
        inputs_shape = inputs.shape #K.array(inputs.shape)
        unpooled_H = (inputs_shape[1]-1)*self.strides[0]+self.filter_size[0]
        unpooled_W = (inputs_shape[2]-1)*self.strides[1]+self.filter_size[1]
        #if type(inputs_shape[0]) is not int: inputs_shape[0] = 1
        max_unpooled = np.zeros((inputs_shape[0], unpooled_H, unpooled_W, inputs_shape[3]))
        for n in range(inputs_shape[0]): # for each example
            for i in range(0, unpooled_H, 2):
                for j in range(0, unpooled_W, 2):
                    for k in range(inputs_shape[2]):
                        if self.indices[n, i//2, j//2, k] == 0:
                            max_unpooled[n, i+0, j+0, k] = inputs[n, i//2, j//2, k]
                        elif self.indices[n, i//2, j//2, k] == 1:
                            max_unpooled[n, i+0, j+1, k] = inputs[n, i//2, j//2, k]
                        elif self.indices[n, i//2, j//2, k] == 2:
                            max_unpooled[n, i+1, j+0, k] = inputs[n, i//2, j//2, k]
                        else: # it is 3
                            max_unpooled[n, i+1, j+1, k] = inputs[n, i//2, j//2, k]
        return max_unpooled
