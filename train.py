import cv2
import os
import numpy as np
from sklearn.svm import SVC
import pickle

hog = cv2.HOGDescriptor()

X = []
y = []

classes = ["car","bike","bus","truck"]

for label, cls in enumerate(classes):
    folder = f"data/{cls}"
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,file))
        if img is None:
            continue

        img = cv2.resize(img,(64,128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = hog.compute(gray)
        X.append(features.flatten())
        y.append(label)

print("Training SVM...")

model = SVC(kernel='linear')
model.fit(X,y)

pickle.dump(model, open("model.pkl","wb"))

print("Model saved!")