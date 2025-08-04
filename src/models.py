from tensorflow.keras.applications import VGG16, VGG19, DenseNet121, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_transfer_model(base_model, input_shape=(224, 224, 3), dropout_rate=0.5, learning_rate=1e-4):
    """
    Generic function to build a transfer learning model.
    :param base_model: Keras base model (without top)
    :param input_shape: input image shape
    :param dropout_rate: dropout rate
    :param learning_rate: learning rate for optimizer
    :return: compiled model
    """
    base_model.trainable = False  # Freeze base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def get_vgg16_model(input_shape=(224, 224, 3)):
    base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    return build_transfer_model(base, input_shape)

def get_vgg19_model(input_shape=(224, 224, 3)):
    base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    return build_transfer_model(base, input_shape)

def get_densenet_model(input_shape=(224, 224, 3)):
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    return build_transfer_model(base, input_shape)

def get_mobilenet_model(input_shape=(224, 224, 3)):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    return build_transfer_model(base, input_shape)
