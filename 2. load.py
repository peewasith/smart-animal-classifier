from pathlib import Path
import numpy as np
import joblib
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.applications import vgg16

classes = ["bird", "cat", "dog"]
paths = []

for c in classes:
    paths.append(Path("dataset") / c)

images = []
labels = []

class_index = 0
for path in paths:
    count_image = 0
    for img in list(path.glob("*.jpeg"))+list(path.glob("*.jpg")):
        img = image.load_img(img,target_size=(224, 224))
        image_array = image.img_to_array(img)
        images.append(image_array)
        labels.append(class_index)
        count_image += 1
    print('The number of ', classes[class_index], count_image)
    class_index += 1

x_train = np.array(images)
y_train = np.array(labels)

# Normalize image data to 0-to-1 range
x_train = vgg16.preprocess_input(x_train)

pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
features_x = pretrained_nn.predict(x_train)
joblib.dump(features_x, "x_train.dat")
joblib.dump(y_train, "y_train.dat")