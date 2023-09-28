import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


# Initialize variables
X = []
y = []

# Load happy faces
for filename in os.listdir('data/joy'):
    img = cv2.imread(os.path.join('data/joy', filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    X.append(img.flatten())
    y.append(1)  # 1 for happy

# Load sad faces
for filename in os.listdir('data/angry'):
    img = cv2.imread(os.path.join('data/angry', filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 50))
    X.append(img.flatten())
    y.append(0)  # 0 for sad

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {round(accuracy * 100,2)}%")


# Load an image file to test, resizing it to 50x50 pixels (grayscale)
img = cv2.imread('PrivateTest_928647.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (50, 50))

# Flatten and reshape the image
img_flatten = img.flatten()
img_flatten = np.reshape(img_flatten, (1, -1))

# Predict the class of the image
result = clf.predict(img_flatten)

# Decode the result
if result[0] == 1:
    print("The image is likely to be a happy face.")
else:
    print("The image is likely to be a sad face.")



