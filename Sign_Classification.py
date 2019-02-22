
# importing the relevant modules
import os
from skimage.filters import gabor
import os
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import skimage
import skimage.filters as filters
from skimage import color
import skimage.filters as filters
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage import feature
from skimage import morphology
from skimage.draw import circle_perimeter
from skimage import img_as_float, img_as_ubyte
from skimage import segmentation as seg
from skimage.morphology import watershed
from scipy import ndimage as nd
from scipy.ndimage import convolve
from skimage import feature
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir('..')
# new_dir = os.getcwd()
os.chdir('data')
data_dir = os.getcwd()
train_data_dir = data_dir + '/asl_alphabet_train/'
test_data_dir = data_dir + '/asl_alphabet_test/'
print(train_data_dir) # test_data_dir)
categories = ['A','B']

# # our data directory  where our sign language images are stored
# DataDir = r"/home/nikunj/Northeastern/ml/ml_project/data/asl_alphabet_train/"
# test_dir = r"/home/nikunj/Northeastern/ml/ml_project/data/asl_alphabet_test/"
# list of sub directory names
categories = ['A','B'] #'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# global img_array  # declaring global variable name for storing read image file

# image size is 50 * 50
Img_Size = 50

training_data = []
test_data_x = []
test_data_y = []
no_of_classes = 10
class_num = np.identity(no_of_classes)


def create_training_data():
    for index, Category in enumerate(categories):
        path = os.path.join(train_data_dir, Category)  # path to cats or dogs Category
        class_num = categories.index(Category)  # 0 = Dog, 1 = Cat
        #selected_row = class_num[:, index]

        # for an img in
        for img in os.listdir(path):
            try:
                # reading the image in gray scale mode and resizing it
                img_array = cv2.imread(os.path.join(path, img))

                # Converting the image to greyscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                # Applying a threshold with Thres Ostu and Thres Binary helps in making the image appear more contrasted
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((5, 5), np.uint8)

                # Removes the white noise in the image
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

                # Dilate gives the exact image Sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=3)

                # Calculates the distance from the center of the object in the image
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

                # Threshold of dist_transform gives you the exact the object
                ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)

                # If we subtract the sure bg and sure fg we get the unknown or the boundary layer
                unknown = cv2.subtract(sure_bg, sure_fg)

                # We connect the unkown with sure fg so that we know for sure that the unkown part is the border
                ret, markers = cv2.connectedComponents(sure_fg)

                # We then add the pixels of the unknown area to 0 to differeniate the borders
                markers = markers + 1
                markers[unknown == 255] = 0

                # We are the joining the markers with original image
                markers = cv2.watershed(img_array, markers)
                img_array[markers == -1] = [255, 0, 0]

                new_array = cv2.resize(img_array, (100, 100))
                gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)

                # We are blurring the image with median filter with kernel of 5*5
                blur_cl1 = cv2.medianBlur(gray, 5)

                # Adding more details in image gaussian image thresholding
                th3 = cv2.adaptiveThreshold(blur_cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                # If the image imbalanced the contrast we are balancing it
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(32, 32))
                cl1 = clahe.apply(th3)

                training_data.append([cl1, class_num])  # creating training data by appending list of new sized images and class it correponds to

            except Exception as e:
                pass

def create_testing_data():
    for index, Category in enumerate(categories):
        path = os.path.join(test_data_dir, Category)
        target_class = categories.index(Category)

        for img in os.listdir(path):
            try:
                # reading the image in gray scale mode and resizing it
                img_array = cv2.imread(os.path.join(path, img))

                # Converting the image to greyscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                # Applying a threshold with Thres Ostu and Thres Binary helps in making the image appear more contrasted
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((5, 5), np.uint8)

                # Removes the white noise in the image
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

                # Dilate gives the exact image Sure background area
                sure_bg = cv2.dilate(opening, kernel, iterations=3)

                # Calculates the distance from the center of the object in the image
                dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

                # Threshold of dist_transform gives you the exact the object
                ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)

                # If we subtract the sure bg and sure fg we get the unknown or the boundary layer
                unknown = cv2.subtract(sure_bg, sure_fg)

                # We connect the unkown with sure fg so that we know for sure that the unkown part is the border
                ret, markers = cv2.connectedComponents(sure_fg)

                # We then add the pixels of the unknown area to 0 to differeniate the borders
                markers = markers + 1
                markers[unknown == 255] = 0

                # We are the joining the markers with original image
                markers = cv2.watershed(img_array, markers)
                img_array[markers == -1] = [255, 0, 0]

                new_array = cv2.resize(img_array, (100, 100))
                gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)

                # We are blurring the image with median filter with kernel of 5*5
                blur_cl1 = cv2.medianBlur(gray, 5)

                # Adding more details in image gaussian image thresholding
                th3 = cv2.adaptiveThreshold(blur_cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                # If the image imbalanced the contrast we are balancing it
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(32, 32))
                cl1 = clahe.apply(th3)
                # new_array = cv2.resize(cl1, (100, 100))
                # gray = cv2.cvtColor(new_array, cv2.COLOR_BGR2GRAY)

                test_data_x.append(cl1)
                test_data_y.append(target_class)
            except Exception as e:
                pass


create_training_data()   # function to create training data
create_testing_data()


random.shuffle(training_data)    # randomly shuffling training data

# for every sample (image, class pair) present in the training data,  print the class labels
for sample in training_data[0:10]:
    print(sample[1])
# create variables to hold training data samples and target values
x = [] #feature
y = [] #label

# seperate out the training data into features and label
for features, label in training_data:
    x.append(features)
    y.append(label)

print(len(x))
print(len(y))
# reshaping the images
x = np.array(x).reshape(-1, 100, 100, 1)  # -1 how many shapes we have


pickle_out  = open("x.pickle", 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out  = open("y.pickle", 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("x.pickle", 'rb')
x = pickle.load(pickle_in)

# finding the tensorflow version
print(tf.__version__)

pickle_in = open("x.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
print(y)

X = X/255.0

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'mse', 'mae', 'cosine', 'mape'])


history = model.fit(X, y, batch_size=30, epochs=4, validation_split=0.3)
print("\nTraining accuracy: ", history.history['acc'])
print("\nCross Validation accuracy: ", history.history['val_acc'])

print(history.history.keys())
test_data_x = np.array(test_data_x).reshape(-1, 100, 100, 1)
ypred = model.predict(x=test_data_x, verbose=1)

hits = [i for i, j in zip(ypred, test_data_y) if i == j]
test_accuracy = len(hits) / len(test_data_y) * 100
print("\nTest data accuracy: ", test_accuracy)


plt.plot(history.history['mean_squared_error'])
plt.show()
plt.plot(history.history['mean_absolute_error'])
plt.show()
plt.plot(history.history['mean_absolute_percentage_error'])
plt.show()
plt.plot(history.history['cosine_proximity'])
plt.show()
plt.plot(history.history['cosine_proximity'])
plt.show()
plt.plot(history.history['val_mean_absolute_error'])
plt.show()
plt.plot(history.history['val_mean_squared_error'])
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()





