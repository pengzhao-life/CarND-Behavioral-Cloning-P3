# Import all modules
import csv 
import cv2
import numpy as np
import math
import sklearn
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split

def readImagesList(file_names):
    """
    Load the list for training images
    """
    lines = []
    for file_name in file_names:
        with open(file_name) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
    return lines
 
def checkData(samples):
    """
    Check samples size
    Note: plt is very slow. so didn't plot
    """
    angles = []
    for sample in samples:
        try:
            angle = float(sample[3])
        except ValueError:
            print("not a float")
            continue
        angles.append(angle)
    
    n_angles = len(angles)
    print("Number of samples =", n_angles)
    
def generator(samples, batch_size=32):
    """
    Data generator. It loads data in batch, to lower memory usage
    """
    num_samples = len(samples)
    # Correct angles for left and right camera
    correction = 0.2
    corrections = [0, correction, -correction]
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                try:
                    center_angle = float(batch_sample[3])
                # It looks like some angle may not be valid float
                # So add this to handle the exception
                except ValueError as e:
                    print ("error: {}, on line {}".format(e,i))
                    continue
                for i in range(3):
                    if 'my_data' in batch_sample[i]:
                        name = batch_sample[i]
                    else:
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    angle = center_angle + corrections[i]
                    angles.append(angle)

            # Flip or mirror image for augmentation
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)
                
            # Covert to numpy array format
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)        
            yield sklearn.utils.shuffle(X_train, y_train)   

# Model from NVidia
def NVIDIAModel():
    """
    Create Nvidia model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def main():
    """
    Load training data, 
    train model with NVIDIA CNN model
    save model as model.h5
    """
    # Load list for training data
    samples = readImagesList(['./data/driving_log.csv',
                              '/opt/carnd_p3/my_data/driving_log.csv']) 

    # Look at data size
    checkData(samples)
    
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # Set our batch size
    batch_size=32
    
    # Compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    
    # Create NVIDIA model
    model = NVIDIAModel()
    
    # Train the model 
    model.compile(loss='mse', optimizer='adam')
    nvidia_model = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)
    
    # Save the model
    model.save('model.h5')
    
    # Print model summary
    model.summary()

if __name__ == '__main__':
   main()

