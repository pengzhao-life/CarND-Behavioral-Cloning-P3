# Behavioral Cloning Project


### This project is to apply deep learning to clone driver's behavior for end to end self driving car.


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/histogram_angles.jpg "Data Histogram"
[image2]: ./output/model_summary.png "Model Summary"
[image3]: ./output/left_2019_09_01_22_54_26_504.jpg "Left Image"
[image4]: ./output/center_2019_09_01_22_54_26_504.jpg "Center Image"
[image5]: ./output/right_2019_09_01_22_54_26_504.jpg "Right Image"
[image7]: ./output/right_2_center.jpg "Right to Center"
[image8]: ./output/flip.png "Flipped Image"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* README.md report, summarizing the results
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* drive.py for driving the car in autonomous mode. (Udacity provided. No change.)

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

This project is based on [NVIDIA Model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). 
The model consists of a convolution neural network with three layers of 5x5 filter, two layers of 3x3 filter, and three fully connected layers (model.py lines 92-110). 
The model includes RELU layers to introduce nonlinearity (code line 99-103), and the data is normalized in the model using a Keras lambda layer (code line 97). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 127). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First, I built a very basic neural network just to verify everything is working.
```
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_tain, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
```
Then I tried the more powerful model LeNet
```
model = Sequential()
# Data normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_tain, y_train, validation_split=0.2, shuffle=True, epochs=5)
model.save('model.h5')
```
Then I look at the even more powerful model NVIDIA model. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 92-110) consisted of a convolution neural network with the following layers and layer sizes.

Use python script `model.summary()` to print out the model summary as below. 
Use `model = load_model('model.h5')` to load saved model if needed.
![alt text][image2]

This is how to understand the output size and parameters number:

The original image is 160x320x3. It turns into 65x320x3 after cropping 70 on the top and 25 on the bottom.

The output size for each layer can be computed as a function of the input volume size (W), 
the receptive field size of the Conv Layer neurons (F), 
the stride with which they are applied (S), 
and the amount of zero padding used (P) on the border, i.e. `(W−F+2P)/S+1`.
[[ref]](http://cs231n.github.io/convolutional-networks/)

The parameters number is:

```filter_size x filter_size x previous_layer_depth x current_layer_depth + current_layer_depth```

Take conv2d_1 for example, the parameters number is `5 x 5 x 3 x 24 + 24 = 1824`

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the overfitting, I add a dropout=0.25.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.


#### 3. Creation of the Training Set & Training Process

I used both the training set provided by Udacity, and the data I recorded myself.

First I check the training set from Udacity. It contains 8036 center images with its steering angle. The histogram is as below.
Each center image also corresponds to one left image and one right image. 
So the total images provided by Udacity are 8036 x 3 = 24108.
![alt text][image1]


To capture good driving behavior, I recorded one lap on track one using center lane driving. Here is an example including left image, center, and right respectively:

Left                       |  Center                   |Right
:-------------------------:|:-------------------------:|:-------------------------:
![alt text][image3]        | ![alt text][image4]       | ![alt text][image5]


I then recorded the vehicle recovering from the left side and right sides of the road back to center 
so that the vehicle would learn to drive back to center. This image show what a recovery looks like starting from off to the right back to center:

Right to Center            |                   
:-------------------------:|
![alt text][image7]        | 

I tried to repeat this process on track two in order to get more data points, but it is very hard to maneuver. So I didn't collect data for track two.

To augment the data sat, I also flipped images and angles. This would add more varieties to the dataset.
For example, here is an image that has then been flipped:

Original                   |  Flipped                   
:-------------------------:|:-------------------------:
![alt text][image4]        | ![alt text][image8]

After the collection process, I had 10491 data points. I then preprocessed this data by scaling and zero mean i.e. `x/255.0 - 0.5`

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Note: training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 5 as evidenced by mean squared error loss of training data and validation data. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
