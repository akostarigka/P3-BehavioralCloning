# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

[//]: # (Image References)

[image1]: ./writeup_figures/NVIDIA_architecture.png "Model Visualization"
[image2]: ./writeup_figures/center_2017_12_06_19_17_24_667.jpg "Center Lane Driving"
[image3]: ./writeup_figures/left_2017_12_06_19_17_24_667.jpg "Left Lane Driving"
[image4]: ./writeup_figures/right_2017_12_06_19_17_24_592.jpg "Right Lane Driving"
[image5]: ./writeup_figures/flipped_image.jpg "Flipped Image"
[image6]: ./writeup_figures/croped_image.jpg "Cropped Image"
[image7]: ./writeup_figures/YUV_image.jpg "YUV Image"
[image8]: ./writeup_figures/Figure_1.png "Train and Validation Loss"


The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md as a writeup report summarizing the results
* video.mp4

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. Network Model Architecture

The Network architecture used for the model is the [NVIDIA Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This approach proved surprisingly powerful since the system learns to drive with minimum training data. The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal.

This model consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The input image is transformed into YUV colorspace and fed into the network. The first layer of the network performs image normalization and mean centering using a Keras lambda layer (model.py line 108). We use strided convolutions in the first three convolutional layers with a 2×2 stride (model.py lines 110-112) and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers (model.py lines 113-114) [[Bojarski et al., *End to End Learning for Self-Driving Cars*]](https://arxiv.org/pdf/1604.07316v1.pdf).

Here is a visualization of the architecture

![alt text][image1]

### 2. Training and Validation Data

To collect data of good driving behavior the Udacity simulator was used in Training Mode. Data was chosen to keep the vehicle driving on the road.

#### Data Creation

Initially two laps were recorded on track one using center lane driving.

| Center Lane Driving  |
|:--------------------:|
| ![alt text][image2] |

In order to increase the number of samples and improve generalization, 2 extra laps were recorder in the opposite direction. "Weak" points (e.g. sharp corners) were repeated multiple times steering abruptly in order to recover the vehicle back to center.

All data were saved in a directory called `Data` including the `driving_log.csv` file and an `IMG` directory of the respective figures.

#### Data Augmentation and Preprocessing

**a. Conversion to YUV**

All images were converted to YUV color space as dictated in the paper ["End to End Learning for Self-Driving Cars"](https://arxiv.org/pdf/1604.07316v1.pdf) that introduced the NVIDIA network architecture used in the project (model.py lines 110-119). In order for the model to work properly, the drive.py file was also changed, including a line that pre-processed the images before they were used in the prediction (drive.py line 65).  

| Original Image   |  YUV Image  |
|:-------------:|:------------------:|
| ![alt text][image2]  |  ![alt text][image7] |

**b. Image flipping**

In order to increase generalization and teach the network how to recover from a poor position or orientation, the data were augmented by adding flipped images and angles.

| Original Image   |  Flipped Image  |
|:-------------:|:------------------:|
| ![alt text][image2]  |  ![alt text][image5] |

**c. Using multiple cameras**

In the project left and right side camera images were also used to train the model. This way, the model was taught how to steer if the car drifts off to the left or the right. During training, all images were fed to the model as if they were coming from the center camera. In order to force the model steer a little harder towards the opposite direction in case of the side images, we used a small correction factor to the actual steering measurement, e.g.  


| Left Image   |  Center Image	| Right Image 		    |
|:-------------:|:-------------------:|:-----------------------:|
| ![alt text][image3]  |  ![alt text][image2] | ![alt text][image4]  |
| Actual Steering Angle + Correction        | Actual Steering Angle |Actual Steering Angle - Correction  				     	|

In our case the correction term was 0.1 (model.py line 25).

**d. Image cropping and resize**

To help the model train faster, we cropped each image to focus on only the portion that was useful for predicting a steering angle. Since the top portion of the image captured distracting elements like trees, hills and sky, and the bottom portion captured the hood of the car, we cropped 60 pixels from the top and 25 pixels from the bottom of each image. The cropping was done inside of the model using a built-in Keras layer (model.py line 104).

| Cropped Image  |
|:--------------------:|
| ![alt text][image6] |

**e. Data normalization and mean centering**

For normalization and mean centering we used a [Keras lambda layer](https://keras.io/layers/core/#lambda) to our model (model.py line 108). Within this layer we normalized the image to a range between 0 and 1 by dividing by 127.5 and mean centered it by subtracting 1.0:
`model.add(Lambda(lambda x: x/127.5-1.0))`.

### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 122). Finally, the data set was randomly shuffled and the training and validation datasets were split as: Training 65% and Validation 35%. In order to reduce overfitting and since the CNN architecture is a very powerful one, the model was trained only for 3 epochs.

Training and Validation Loss after 3 epochs can be seen in the following figure.

![alt text][image8]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

In case the vehicle fell off the track, to improve driving behavior, the "weak" spot was repeated in Training Mode trying to veer off to the side and recover back to center.

Finally the vehicle was able to successfully stay on the road during the whole route of the First Track.

### 4. Conclusion

In this project we have used the Behavioral Cloning technique in order to successfully drive a vehicle on a test track. We have collected training data using the Udacity simulator, trained the NVIDIA CNN and evaluated the network’s performance in simulation. The training and validation losses as well as the vehicle performance on an output video illustrates the effectiveness of the approach.
