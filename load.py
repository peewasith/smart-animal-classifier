from pathlib import Path
import numpy as np
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

classes = ["bird", "cat", "dog"]
paths = [Path("dataset") / c for c in classes]

images = []
labels = []

for class_index, path in enumerate(paths):
    count_image = 0
    for img_path in list(path.glob("*.jpeg")) + list(path.glob("*.jpg")):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(class_index)
        count_image += 1
    print(f"The number of {classes[class_index]} images: {count_image}")

x_train = np.array(images)
y_train = np.array(labels)

# Normalize images for VGG16
x_train = vgg16.preprocess_input(x_train)

# Load pretrained VGG16
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features
features_x = pretrained_nn.predict(x_train)

# Save features and labels
joblib.dump(features_x, "x_train.dat")
joblib.dump(y_train, "y_train.dat")

print("Feature extraction completed and saved!")
