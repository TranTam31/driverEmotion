# # models/cnn.py

# import tensorflow as tf
# from tensorflow.keras.layers import Activation, Conv2D, Dropout
# from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Flatten, Input, MaxPooling2D
# from tensorflow.keras.layers import SeparableConv2D, Add
# from tensorflow.keras.regularizers import l2


# def simple_CNN(input_shape, num_classes):
#     """Simple CNN architecture with multiple convolutional blocks"""
#     model = Sequential()
#     model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same',
#                      name='image_array', input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(0.5))

#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
#     model.add(GlobalAveragePooling2D())
#     model.add(Activation('softmax', name='predictions'))
#     return model


# def simpler_CNN(input_shape, num_classes):
#     """A simpler CNN with stride-based downsampling"""
#     model = Sequential()
#     model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same',
#                      name='image_array', input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=16, kernel_size=(5, 5),
#                      strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=32, kernel_size=(5, 5),
#                      strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=64, kernel_size=(3, 3),
#                      strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(filters=64, kernel_size=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3),
#                      strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=128, kernel_size=(3, 3),
#                      strides=(2, 2), padding='same'))

#     model.add(Conv2D(filters=256, kernel_size=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(filters=num_classes, kernel_size=(3, 3),
#                      strides=(2, 2), padding='same'))

#     model.add(Flatten())
#     model.add(Activation('softmax', name='predictions'))
#     return model


# def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
#     """Tiny Xception architecture with residual connections"""
#     regularization = l2(l2_regularization)

#     # base
#     img_input = Input(input_shape)
#     x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                use_bias=False)(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     # module 1
#     residual = Conv2D(8, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(8, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(8, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     # module 2
#     residual = Conv2D(16, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     # module 3
#     residual = Conv2D(32, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     # module 4
#     residual = Conv2D(64, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     x = Conv2D(num_classes, (3, 3), padding='same')(x)
#     x = GlobalAveragePooling2D()(x)
#     output = Activation('softmax', name='predictions')(x)

#     model = Model(img_input, output)
#     return model


# def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
#     """Mini Xception architecture with residual connections"""
#     regularization = l2(l2_regularization)

#     # base
#     img_input = Input(input_shape)
#     x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                use_bias=False)(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     # module 1
#     residual = Conv2D(16, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     # Removed kernel_regularizer parameter
#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     # module 2
#     residual = Conv2D(32, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     # Removed kernel_regularizer parameter
#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     # module 3
#     residual = Conv2D(64, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     # Removed kernel_regularizer parameter
#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     # module 4
#     residual = Conv2D(128, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     # Removed kernel_regularizer parameter
#     x = SeparableConv2D(128, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(128, (3, 3), padding='same',
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     x = Conv2D(num_classes, (3, 3), padding='same')(x)
#     x = GlobalAveragePooling2D()(x)
#     output = Activation('softmax', name='predictions')(x)

#     model = Model(img_input, output)
#     return model

# def big_XCEPTION(input_shape, num_classes):
#     """Big Xception architecture with residual connections"""
#     img_input = Input(input_shape)
#     x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
#     x = BatchNormalization(name='block1_conv1_bn')(x)
#     x = Activation('relu', name='block1_conv1_act')(x)
#     x = Conv2D(64, (3, 3), use_bias=False)(x)
#     x = BatchNormalization(name='block1_conv2_bn')(x)
#     x = Activation('relu', name='block1_conv2_act')(x)

#     residual = Conv2D(128, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block2_sepconv1_bn')(x)
#     x = Activation('relu', name='block2_sepconv2_act')(x)
#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block2_sepconv2_bn')(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])

#     residual = Conv2D(256, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = Activation('relu', name='block3_sepconv1_act')(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block3_sepconv1_bn')(x)
#     x = Activation('relu', name='block3_sepconv2_act')(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block3_sepconv2_bn')(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = Add()([x, residual])
#     x = Conv2D(num_classes, (3, 3), padding='same')(x)
#     x = GlobalAveragePooling2D()(x)
#     output = Activation('softmax', name='predictions')(x)

#     model = Model(img_input, output)
#     return model


# if __name__ == "__main__":
#     input_shape = (64, 64, 1)
#     num_classes = 7
#     # model = tiny_XCEPTION(input_shape, num_classes)
#     # model.summary()
#     # model = mini_XCEPTION(input_shape, num_classes)
#     # model.summary()
#     # model = big_XCEPTION(input_shape, num_classes)
#     # model.summary()
#     model = simple_CNN((48, 48, 1), num_classes)
#     model.summary()

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os
from tensorflow.keras import layers, Model, Sequential
import numpy as np

class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation Block (SE) as shown in Image 1
    """
    def __init__(self, in_channels, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Define layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(in_channels // reduction_ratio)
        self.relu = layers.ReLU()
        self.fc2 = layers.Dense(in_channels)
        self.sigmoid = layers.Activation('sigmoid')
        self.reshape = layers.Reshape((1, 1, in_channels))
        self.multiply = layers.Multiply()
        
    def call(self, inputs):
        # Squeeze
        x = self.global_pool(inputs)
        
        # Excitation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        # Scale
        x = self.reshape(x)
        output = self.multiply([inputs, x])
        
        return output
    
    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            "in_channels": self.in_channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config

class MSEBlock(layers.Layer):
    """
    Modified Squeeze-and-Excitation Block (MSE) as shown in Image 1
    Contains additional FC and ReLU layers compared to SE
    """
    def __init__(self, in_channels, reduction_ratio=16, **kwargs):
        super(MSEBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Define layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(in_channels // reduction_ratio)
        self.relu1 = layers.ReLU()
        self.fc2 = layers.Dense(in_channels)  # Extra FC
        self.relu2 = layers.ReLU()  # Extra ReLU
        self.fc3 = layers.Dense(in_channels)
        self.sigmoid = layers.Activation('sigmoid')
        self.reshape = layers.Reshape((1, 1, in_channels))
        self.multiply = layers.Multiply()
        
    def call(self, inputs):
        # Squeeze
        x = self.global_pool(inputs)
        
        # Extended Excitation
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        # Scale
        x = self.reshape(x)
        output = self.multiply([inputs, x])
        
        return output
    
    def get_config(self):
        config = super(MSEBlock, self).get_config()
        config.update({
            "in_channels": self.in_channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config

class ILABBlock(layers.Layer):
    """
    ILAB Block as shown in Image 2
    Combines MSE with additional processing
    """
    def __init__(self, in_channels, reduction_ratio=16, **kwargs):
        super(ILABBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Left branch - MSE
        self.mse = MSEBlock(in_channels, reduction_ratio)
        
        # Right branch - Global Encoder
        self.max_pool = layers.GlobalMaxPooling2D()
        self.reshape_pool = layers.Reshape((1, 1, in_channels))
        self.mse_right = MSEBlock(in_channels, reduction_ratio)
        self.conv_transpose = layers.Conv2DTranspose(
            in_channels, kernel_size=1, strides=1, padding='valid'
        )
        
        # Addition for both branches
        self.add = layers.Add()
        
    def call(self, inputs):
        # Left branch
        x_c = self.mse(inputs)
        
        # Right branch
        x_e = self.max_pool(inputs)
        x_e = self.reshape_pool(x_e)
        x_e = self.mse_right(x_e)
        x_e = self.conv_transpose(x_e)
        
        # Combine branches through addition
        # Ensure shapes match for addition
        output = self.add([x_c, x_e])
        
        return output
    
    def get_config(self):
        config = super(ILABBlock, self).get_config()
        config.update({
            "in_channels": self.in_channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config

def create_complete_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Complete model architecture as shown in Image 3
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    x = layers.Resizing(192, 192)(inputs)
    
    # Pre-processing
    x = layers.Conv2D(32, kernel_size=7, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # First convolutional block
    # Commented out - so we keep 32 channels
    # x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.ReLU()(x)
    x = MSEBlock(32)(x)  # Changed from 128 to 32
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MSEBlock(64)(x)  # Changed from 128 to 64
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # First main block
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MSEBlock(64)(x)  # Changed from 512 to 64
    x = ILABBlock(64)(x)  # Changed from 512 to 64
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Second main block
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MSEBlock(128)(x)  # Changed from 512 to 64
    x = ILABBlock(128)(x)  # Changed from 512 to 64
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Third main block
    x = layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = MSEBlock(256)(x)  # Changed from 512 to 64
    x = ILABBlock(256)(x)  # Changed from 512 to 64
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Dropout(0.2)(x)
    
    # Classifier
    x = layers.GlobalAveragePooling2D()(x)  # Pooling
    x = layers.Flatten()(x)  # Flatten
    x = layers.Dense(256)(x)  # FC
    x = layers.ReLU()(x)  # ReLU
    x = layers.Dropout(0.6)(x)  # Dropout
    x = layers.Dense(128)(x)  # FC
    x = layers.ReLU()(x)  # ReLU
    x = layers.Dense(num_classes, activation='softmax')(x)  # FC with Softmax
    
    # Create model
    model = Model(inputs=inputs, outputs=x, name="CompleteModel")
    
    return model

if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    # model = tiny_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = mini_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = big_XCEPTION(input_shape, num_classes)
    # model.summary()
    model = create_complete_model((48, 48, 1), num_classes)
    model.summary()