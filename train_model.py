from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import os

width, height = 28, 28
train_dir = "train"
test_dir = "test"

train_samples = [len(os.listdir("train/"+str(x))) for x in range(10)]
train_samples = sum(train_samples)

test_samples = len(os.listdir("test/"))

epoch = 30

model = Sequential()
model.add(Conv2D(16, 5, 5, activation = 'relu', input_shape = (width, height, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, 5, 5, activation = 'relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (width, height),
    batch_size = 32,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory(
     test_dir,
     target_size = (width,height),
     batch_size = 32,
     class_mode = 'categorical'
)

model.fit_generator(
    train_generator,
    samples_per_epoch = train_samples,
    nb_epoch = epoch,
    validation_data = test_generator,
    nb_val_samples = test_samples
)

model.save('mnist_cnn.model')
