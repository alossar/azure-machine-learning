import os
import glob
import matplotlib.pyplot as plt

from azureml.core import Run

from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.utils import to_categorical
from keras.callbacks import Callback

import tensorflow as tf

# start an Azure ML run
run = Run.get_context()

data_folder = os.environ['concretedata']
print('training dataset is stored here:', data_folder)

train_path = glob.glob(os.path.join(data_folder, '**/train'), recursive=True)[0]
valid_path = glob.glob(os.path.join(data_folder, '**/valid'), recursive=True)[0]
test_path = glob.glob(os.path.join(data_folder, '**/test'), recursive=True)[0]

num_classes = 2
image_resize = 224

batch_size_training = 100
batch_size_validation = 50

data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = data_generator.flow_from_directory(
    train_path,
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical'
)

validation_generator = data_generator.flow_from_directory(
    directory= valid_path,
    target_size=(image_resize, image_resize),    
    batch_size=batch_size_validation,
    class_mode="categorical"
)

test_generator = data_generator.flow_from_directory(
    directory= test_path,
    target_size=(image_resize, image_resize),
    batch_size=100,
    shuffle=False,
    class_mode="categorical")

steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 1

model = Sequential()
model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable = False

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=[LogRunMetrics()]
)

score = model.evaluate_generator(test_generator, verbose=0)

run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('Concrete with Keras MLP ({} epochs)'.format(num_epochs), fontsize=14)
plt.plot(history.history['val_accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['val_loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")