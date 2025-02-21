# data_modelling.py

from tensorflow.keras.layers import ConvLSTM2D, MaxPooling3D, TimeDistributed, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM
from tensorflow.keras.models import Sequential

class ModelBuilder:
    @staticmethod
    def create_LRCN_model(sequence_length, image_height, image_width, num_channels, num_classes):
        model = Sequential([
            TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),
                            input_shape=(sequence_length, image_height, image_width, num_channels)),
            TimeDistributed(MaxPooling2D((4, 4))),
            TimeDistributed(Dropout(0.25)),
            TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')),
            TimeDistributed(MaxPooling2D((4, 4))),
            TimeDistributed(Dropout(0.25)),
            TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Dropout(0.25)),
            TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')),
            TimeDistributed(MaxPooling2D((2, 2))),
            TimeDistributed(Flatten()),
            LSTM(32),
            Dense(num_classes, activation='softmax')
        ])

        model.summary()
        return model
