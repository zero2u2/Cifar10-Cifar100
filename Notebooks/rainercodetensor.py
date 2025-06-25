

# %%
# TensorFlow + Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
print(scipy.__version__)


# Standard-Pakete
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import logging

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Verwende nur die erste GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU aktiviert:", gpus[0])
    except RuntimeError as e:
        print("❌ Fehler beim Setzen der GPU:", e)
        
# Scikit-learn (für Klassifikationsauswertung)
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
print("TensorFlow-Version:", tf.__version__)
print("Verfügbare GPUs:", tf.config.list_physical_devices('GPU'))
# CIFAR-10 Dataset laden
from tensorflow.keras.datasets import cifar10


# %%
# Fetch "Fashion MNIST" data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255



# Map for human readable class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

class_names_map = {i:class_names[i] for i in range(len(class_names))}

# %% [markdown]
# Costum implementation for Swish acivation function
# 
# ---
# Official Papaer: https://arxiv.org/pdf/1710.05941v1.pdf
# 
# 

# %%
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import swish

model = keras.Sequential([
    layers.Dense(128, activation=swish),
    layers.Dense(10, activation='softmax')
])

# %% [markdown]
# Custom implementation for Mish activation function
# 
# ---
# Official Paper: https://arxiv.org/pdf/1908.08681.pdf
# 

# %%
import tensorflow as tf

def mish(x):
    return x * tf.math.tanh(tf.nn.softplus(x))

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Input(shape=(32, 32, 3)),
    Flatten(),
    Dense(128, activation=mish),
    Dense(10, activation='softmax')
])



# %%
num_classes = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# %%
from tensorflow_addons.activations import mish  # <== Stelle sicher, dass das importiert ist

def define_my_cifar10_model():
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation=mish, padding="same", input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation=mish, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation=mish,
                           kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation=mish,
                           kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation="softmax"))
    return model

# %%
my_cifar10_model = define_my_cifar10_model()
my_cifar10_model.summary()


# %%
data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=20,  # randomly rotate images in the range
        zoom_range = 0, # Randomly zoom image
        shear_range=0, # Shear angle in counter-clockwise direction in degrees
        width_shift_range=0,  # randomly shift images horizontally
        height_shift_range=0,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False, # randomly flip images
        validation_split=0.2) # Part of training used as validation 

# %%
checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath="/content/drive/MyDrive/Pattern_Recognition/my_cifar10_net.keras",
    verbose=1,
    save_best_only=True
)

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=16,
    restore_best_weights=True
)

logdir = os.path.join(
    "/content/drive/MyDrive/Pattern_Recognition/logs_cifar_10",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


# %%
# Compile the model
import scipy
from tensorflow.keras import backend as K
my_cifar10_model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['acc'])
batch_s = 32

# data_generator.fit(x_train)

history = my_cifar10_model.fit(data_generator.flow(x_train, y_train, batch_size=batch_s),
                      validation_data = data_generator.flow(x_train, y_train,batch_size=batch_s, subset='validation'),
         callbacks=[earlystopping,checkpoints,reduce_lr,tensorboard_callback],
                    verbose=1,epochs=250, batch_size=batch_s, shuffle=True)    

# %%
predictions_my_cifar10=my_cifar10_model.predict(x_test) 
predicted_classes_my_cifar10=np.argmax(predictions_my_cifar10,axis=1)
y_test_values = np.argmax(y_test,axis=-1) # There are in categorical form(One hot encoded) 
print(classification_report(y_test_values, predicted_classes_my_cifar10, target_names=class_names,digits=5))

# %%
y_test_names = [class_names_map[i] for i in y_test_values]
predict_names_my_cifar10 = [class_names_map[i] for i in predicted_classes_my_cifar10]
fig, axs = plt.subplots(1,1,figsize=(10,10),dpi=100)
ConfusionMatrixDisplay.from_predictions(y_test_names, predict_names_my_cifar10, labels=class_names,ax=axs)

# %%
# Load the TensorBoard notebook extension
%load_ext tensorboard

# %%
%tensorboard --logdir "/content/drive/MyDrive/Pattern_Recognition/logs_cifar_10"


