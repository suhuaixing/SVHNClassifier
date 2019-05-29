# SVHNClassifier
A TensorFlow 2.0a implementation of Single-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks 

### Prerequisites:

TensorFlow 2.0-alpha

Or

TensorFlow-GPU 2.0-alpha

cuDNN>=7.4.1 （If using GPU version）

### Data:

4200 image files

28*28 pixels

70% for training (shuffled)

30% for testing (keep order)

### Code:

using TensorFlow **tf.function**, a new way to run graph on GPU

Result:

200 epochs:

training acc: 99.0%

testing acc: 93.1%


### Misc:
**Acc**: 

acc.csv

**Model**:

./checkpoints

**Predictions**:

test_resultXXXX.csv



If using **Google Colab**, there is a Colab version code.

it will save files on your **Google Drive**.

