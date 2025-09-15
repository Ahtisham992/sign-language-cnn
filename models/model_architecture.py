"""
CNN Model Architectures for Sign Language Digits Recognition
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                                     Dense, BatchNormalization, Input,
                                     GlobalAveragePooling2D)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from config import *


class CNNModels:
    @staticmethod
    def create_basic_cnn(input_shape=(64, 64, 3), num_classes=10, dropout_rate=0.3,
                         l1_lambda=0.001, l2_lambda=0.001):
        """
        Create a basic CNN model
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def create_advanced_cnn(input_shape=(64, 64, 3), num_classes=10, dropout_rate=0.3,
                            l1_lambda=0.001, l2_lambda=0.001):
        """
        Create an advanced CNN model with batch normalization
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate * 0.5),

            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate * 0.5),

            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate * 0.5),

            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.5),

            # Dense Layers
            GlobalAveragePooling2D(),
            Dense(256, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Dropout(dropout_rate),
            Dense(128, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def create_deep_cnn(input_shape=(64, 64, 3), num_classes=10, dropout_rate=0.3,
                        l1_lambda=0.001, l2_lambda=0.001):
        """
        Create a deeper CNN model inspired by VGG architecture
        """
        model = Sequential([
            # Block 1
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   input_shape=input_shape,
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.25),

            # Block 2
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.25),

            # Block 3
            Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.5),

            # Block 4
            Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.5),

            # Classifier
            Flatten(),
            Dense(1024, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Dropout(dropout_rate),
            Dense(512, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda)),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def create_residual_block(x, filters, kernel_size=3, stride=1):
        """
        Create a residual block
        """
        shortcut = x

        # Main path
        x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)

        # Shortcut path
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride)(shortcut)
            shortcut = BatchNormalization()(shortcut)

        # Add shortcut
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)

        return x

    @staticmethod
    def create_resnet_like(input_shape=(64, 64, 3), num_classes=10, dropout_rate=0.3,
                           l1_lambda=0.001, l2_lambda=0.001):
        """
        Create a ResNet-like model
        """
        inputs = Input(shape=input_shape)

        # Initial convolution
        x = Conv2D(64, 7, strides=2, padding='same',
                   kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda))(inputs)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)

        # Residual blocks
        x = CNNModels.create_residual_block(x, 64)
        x = CNNModels.create_residual_block(x, 64)

        x = CNNModels.create_residual_block(x, 128, stride=2)
        x = CNNModels.create_residual_block(x, 128)

        x = CNNModels.create_residual_block(x, 256, stride=2)
        x = CNNModels.create_residual_block(x, 256)

        x = CNNModels.create_residual_block(x, 512, stride=2)
        x = CNNModels.create_residual_block(x, 512)

        # Global average pooling and classification
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(256, activation='relu',
                  kernel_regularizer=l1_l2(l1=l1_lambda, l2=l2_lambda))(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model

    @staticmethod
    def compile_model(model, learning_rate=0.001, optimizer_type='adam'):
        """
        Compile the model with specified optimizer and learning rate
        """
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    @staticmethod
    def get_model_summary(model):
        """
        Get detailed model summary
        """
        return model.summary()

    @staticmethod
    def count_parameters(model):
        """
        Count total and trainable parameters
        """
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }