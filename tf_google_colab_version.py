from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import tensorflow as tf
import numpy as np
from pathlib import Path
import csv
import os
# if in google colab, uncomment the three code below
from google.colab import drive
drive.mount('/content/gdrive')

!pip install tf-nightly-gpu-2.0-preview

# for image loading performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
# download data (4MB)
IMAGE_URL = "https://raw.githubusercontent.com/suhuaixing/SVHNClassifier/master/image.zip"
image_file = tf.keras.utils.get_file("images.zip", IMAGE_URL, extract=True)
# zip file path
image_file_path = Path(image_file)
p = image_file_path.parent/"image"
# create a file list
all_image_paths = list(p.glob('*.jpg'))

all_image_paths = sorted(
    all_image_paths, key=lambda x: int(x.stem))
# pathlib return an object, we convert it to string
all_image_paths_str = []
for path in all_image_paths:
    all_image_paths_str.append(str(path))
# sort it by number

# load label file
LABELS_URL = "https://raw.githubusercontent.com/suhuaixing/SVHNClassifier/master/labels.csv"

label_file = tf.keras.utils.get_file("labels.csv", LABELS_URL)

# read labels
label_names = ['0.0', '1.0', '2.0', '3.0',
               '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
all_image_labels = []
with open(label_file, newline='') as f:
    for row in csv.reader(f):
        all_image_labels.append(int(float(row[0])))


IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28


def preprocess_image(image):
    image = tf.io.decode_jpeg(image, channels=3)
    # a must, since decode_jpeg return uint8, we need float32
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# create dataset

TRAIN_SIZE = 2940  # 70% for training
TEST_SIZE = 1260

# file paths and labels
train_image_paths = all_image_paths_str[:TRAIN_SIZE]
train_image_labels = all_image_labels[:TRAIN_SIZE]
test_image_paths = all_image_paths_str[TRAIN_SIZE:]
test_image_labels = all_image_labels[TRAIN_SIZE:]
# training dataset
train_image_ds = tf.data.Dataset.from_tensor_slices(train_image_paths).map(
    load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
train_label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(train_image_labels, tf.int64))
train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))

# testing dataset
test_image_ds = tf.data.Dataset.from_tensor_slices(test_image_paths).map(
    load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(test_image_labels, tf.int64))
test_image_label_ds = tf.data.Dataset.zip((test_image_ds, test_label_ds))

# create model

BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.

train_ds = train_image_label_ds.cache()
train_ds = train_ds.shuffle(buffer_size=TRAIN_SIZE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.


test_ds = test_image_label_ds.cache()
#test_ds = test_ds.shuffle(buffer_size=TEST_SIZE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')

# Categorical Loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

# using tensorflow 2.0 new feature: Eager function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, epoch):
    predictions = model(images)

    # if you need to save the prediction
    if (epoch % SAVE_STEP == 0):
        tf.print(predictions, summarize=TEST_SIZE,
                                  output_stream="file://"+"/content/gdrive/My Drive/Colab Notebooks/"+"test_value"+str(epoch)+".log")
                #  output_stream="file://"+"test_value"+str(epoch)+".log")
    #
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

# create model


class TrainModel(Model):
    def __init__(self):
        super(TrainModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')
        # self.save

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = TrainModel()


# in google colab, 200 epochs for 10 minutes.
# model is good enough. 99% acc.
EPOCHS = 2000
# save the model and prediction
SAVE_STEP = 50


def write_to_csv(epoch):
    # with open("test_value"+str(epoch)+".log") as infile:
    with open("/content/gdrive/My Drive/Colab Notebooks/test_value"+str(epoch)+".log") as infile:
        arr2D = np.fromstring(infile.read().replace(
            "[", "").replace("]", ""), sep=" ").reshape(-1, 10)
        # Get the maximum values of each row i.e. along axis 1
        maxInRows = np.argmax(arr2D, axis=1)
        np.savetxt('/content/gdrive/My Drive/Colab Notebooks/test_result'+str(epoch)+'.csv',
        # np.savetxt('test_result'+str(epoch)+'.csv',
                   maxInRows, delimiter='/n', fmt='%1.0f')


with open("/content/gdrive/My Drive/Colab Notebooks/acc.csv", "w") as f:
# with open("acc.csv", "w") as f:
    f.write(f"epoch,train_loss,train_accuracy,test_loss.result,test_accuracy\n")


for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels, epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
    # if you need to save the acc and loss or on Google Drive
    # with open("acc.csv", "a+") as f:
    with open("/content/gdrive/My Drive/Colab Notebooks/acc.csv", "a+") as f:
        f.write(f"{epoch+1},{train_loss.result()},{train_accuracy.result()*100},{test_loss.result()},{test_accuracy.result()*100}\n")

    if(epoch % SAVE_STEP == 0):
        # if you need to save the model
        # model.save_weights(f'./checkpoints/my_checkpoint{epoch:04d}')
        # if you want to save it to Google Drive, use code below
        model.save_weights(
            f'/content/gdrive/My Drive/Colab Notebooks/checkpoints/checkpoint{epoch:04d}')
        print('export model: ', epoch)
        # if you need to save the prediction
        write_to_csv(epoch)
