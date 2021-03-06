#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cnn"""

__author__ = 'yp'

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# 自定义loss
# 方法1
# def losses(labels: list, preds: list):
#     l = 0
#     for i in range(len(labels)):
#         l += tf.reduce_sum(((labels[i] - preds[i]) ** 2) * (i + 1))
#     return l
# model.add_loss(losses([label_1, label_2], [pred_1, pred_2]))
# model.compile('adam')

# 方法2

# def custom_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))
# model.compile(loss=custom_loss)

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)

new_model = Model(inputs=model.input,
                  outputs=model.get_layer('dense').get_output_at(0))
new_output = new_model.predict(test_images)

print(new_output, new_output.shape)
