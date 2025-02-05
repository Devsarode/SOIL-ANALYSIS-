from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

input_shape = (224, 224, 3)
classes = ['Alluvial Soil', 'Clayey Soil', 'Laterite Soil']

img_height, img_width = 256, 256
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes = classes,
    class_mode='categorical'
)
  #'binary' or 'categorical'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

validation_generator = val_datagen.flow_from_directory(
  
    'C:\Users\aman0\OneDrive\Desktop\new_ML_MODEL\valid',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    classes = classes,
    class_mode='categorical'
)





import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import recall_score, f1_score
import re
from sklearn.metrics import confusion_matrix

mobilenet_v2 = MobileNetV2(weights='imagenet', include_top=False)


x = mobilenet_v2.output
x = GlobalAveragePooling2D()(x)  # Pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(len(classes), activation='softmax')(x)  # Output layer

# Create the final model
model = tf.keras.Model(inputs=mobilenet_v2.input, outputs=predictions)

# print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',  # Standard metric
                       tf.keras.metrics.Recall()])


results = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100,
    verbose=1)

plt.figure(figsize=(10, 6))
plt.plot(results.history['accuracy'], label='Training Accuracy')
plt.plot(results.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation recall
plt.figure(figsize=(10, 6))
plt.plot(results.history['recall'], label='Training Recall')
plt.plot(results.history['val_recall'], label='Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Recall')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(results.history['loss'], label='Training Loss')
plt.plot(results.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


model.evaluate(validation_generator)
from keras.models import load_model
model.save('model_soil.h5') 
tf.keras.utils.plot_model(model, to_file='model_layers.png', show_shapes=True, show_layer_names=True)