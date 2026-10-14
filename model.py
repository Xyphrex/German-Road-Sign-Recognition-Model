import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


training_data = './GTSRB Train/GTSRB/Final_Training/Images'
test_ground_truths = pd.read_csv("./COMP3419-A2-Opt3-Evaluation/GTSRB_Test_GT.csv", sep=';')
test_image_folder = "./GTSRB Test/Final_Test/Images"


gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Detected GPU's: {gpus}")

def load_gtsrb_data(training_data):
    images = []
    labels = []

    for image_folder in os.listdir(training_data):
        image_folder_path = os.path.join(training_data, image_folder)
        if os.path.isdir(image_folder_path):
            label = int(image_folder)

            for image_file in os.listdir(image_folder_path):
                if image_file.endswith('.ppm'):
                    image_path = os.path.join(image_folder_path, image_file)
                    original_img = cv2.imread(image_path)
                    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    img = img / 255.0
                    img = cv2.resize(img, (32, 32))
                  
                    images.append(img)
                    labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    # labeldict = {}
    # print(len(images))
    # print(len(labels))
    # for i in labels:
    #     if i not in labeldict:
    #         labeldict[int(i)] = 1
    #     else:
    #         labeldict[i] += 1
    # print(labeldict)

    return images, labels

def load_test_data(test_ground_truths, test_image_folder):
    test_images = []
    test_labels = []
    images = os.listdir(test_image_folder)
    images = [f for f in images if f.endswith('.ppm')]
    images = sorted(images, key=lambda x: int(x.split('.')[0]))

    for image_index in range(len(images)):
        if images[image_index].endswith(".ppm"):
            image_path = os.path.join(test_image_folder, images[image_index])
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = cv2.resize(img, (32, 32))
            test_images.append(img)
            test_labels.append(test_ground_truths.iloc[image_index]["ClassId"])
        
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return test_images, test_labels


training_images, training_labels = load_gtsrb_data(training_data)
test_images, test_labels = load_test_data(test_ground_truths, test_image_folder)

# print(test_labels[0])
# plt.imshow(test_images[0])
# plt.show()
# quit()

training_images, validation_images, training_labels, validation_labels = train_test_split(training_images, training_labels, test_size=0.2, random_state=42)

print("Training data shape:", training_images.shape)
print("Validation data shape:", validation_images.shape)
print("Test data shape:", test_images.shape)

'''
Old Model as referenced in the assignment report
'''

# cnn_model = tf.keras.Sequential()
# cnn_model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu'))
# cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# cnn_model.add(tf.keras.layers.Dropout(0.2))
# cnn_model.add(tf.keras.layers.Flatten())
# cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
# cnn_model.add(tf.keras.layers.Dropout(0.3))
# cnn_model.add(tf.keras.layers.Dense(43, activation='softmax'))

cnn_model = tf.keras.Sequential()
cnn_model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
cnn_model.add(tf.keras.layers.BatchNormalization())
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
cnn_model.add(tf.keras.layers.BatchNormalization())
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
cnn_model.add(tf.keras.layers.BatchNormalization())
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(tf.keras.layers.Dropout(0.5))
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
cnn_model.add(tf.keras.layers.Dropout(0.5))
cnn_model.add(tf.keras.layers.Dense(43, activation='softmax'))

cnn_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])

history = cnn_model.fit(training_images, training_labels, batch_size=64, epochs=100, verbose=1, validation_data=(validation_images, validation_labels))

score = cnn_model.evaluate(test_images, test_labels, verbose=0)
print('Test Accuracy: {:.4f}'.format(score[1]))

cnn_model.save('./road_sign_model.keras')

plt.plot(history.history["accuracy"], label='Training Accuracy')
plt.plot(history.history["val_accuracy"], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.savefig('model_accuracy.png')
plt.show()