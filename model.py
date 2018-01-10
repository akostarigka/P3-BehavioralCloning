import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras import backend as K

# Load the training data taken from the simulator in Training Mode
samples = []
with open('./Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split the data in training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.35)

# Define correction factor
correction = 0.1


# Define generator with an extra argument to distinguish between training and prediction mode
def generator(samples, batch_size=32, mode="prediction"):
    # In case of training feed center, left and right camera images to the model
    if mode == "training":
        img_per_row = 3
    # In case of prediction use only center camera images
    else:
        img_per_row = 1

    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        # Shuffle the training data
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            # Iterate through the batch samples
            for batch_sample in batch_samples:
                for i in range(img_per_row):
                    # Extract path for each camera image
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './Data/IMG/' + filename
                    # Read in images
                    image = mpimg.imread(current_path)
                    # Convert input image to YUV (optimal format for NVIDIA architecture)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                    # Append image to the list of images
                    images.append(image)
                    # Create adjusted steering measurements for the side camera images
                    if i == 0:
                        angle = float(batch_sample[3])
                    elif i == 1:
                        angle = float(batch_sample[3]) + correction
                    elif i == 2:
                        angle = float(batch_sample[3]) - correction
                    # Append steering measurements to the list of angles
                    angles.append(angle)
                    if mode == "training":  # In case of training
                        # Augment training data
                        augmented_images, augmented_angles = [], []
                        for image, angle in zip(images, angles):
                            # Appent actual images and respective angles
                            augmented_images.append(image)
                            augmented_angles.append(angle)
                            # Append flipping images taking the opposite
                            # sign of the steering measurement
                            augmented_images.append(cv2.flip(image, 1))
                            augmented_angles.append(angle*-1.0)

            if mode == "training":  # In case of training use the augmented data set
                # Convert the images and steering measurements to numpy arrays
                # to be used in keras
                X_train = np.array(augmented_images)
                y_train = np.array(augmented_angles)
            else:  # In case of prediction use the actual data set
                # Convert the images and steering measurements to numpy arrays
                # to be used in keras
                X_train = np.array(images)
                y_train = np.array(angles)

            # Shuffle the data
            yield sklearn.utils.shuffle(X_train, y_train)


# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, mode="training")
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
# Add cropping layer to choose only the useful image portion
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
# Resise images
model.add(Lambda(lambda x: K.tf.image.resize_images(x, size=(66, 200))))
# Preprocess data: normalize and mean center
model.add(Lambda(lambda x: x/127.5-1.0))
# Employ Network architecture: NVIDIA
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))

# Compile model: loss function = mean squared error, optimizer = adam
model.compile(loss='mse', optimizer='adam')
# Create history object that contains the training and validation loss for each epoch
history_obj = model.fit_generator(train_generator,
                                  samples_per_epoch=len(train_samples)*6,
                                  validation_data=validation_generator,
                                  nb_val_samples=len(validation_samples),
                                  nb_epoch=3)

# Print the keys contained in the history object
print(history_obj.history.keys())

# Save the train model
model.save('model.h5')

# Plot the training and validation loss for each epoch
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation_set'], loc='upper right')
plt.show()
