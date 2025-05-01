import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Debug: Check class folders
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
train_dir = '/Users/anthonyjirano/Desktop/CSUF/CSUF Spring 2025/AI/Project Testing/skincare/data/train'
print("Training classes:", sorted(os.listdir(train_dir)))

# 1. Load data with validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',  # Must be 'categorical' for 3+ classes
    subset='training'
)

# Debug: Verify output shape
for x, y in train_generator:
    print("Batch label shape:", y.shape)  # Should be (batch_size, 3)
    print("Unique labels in batch:", np.unique(np.argmax(y, axis=1)))
    break

# 2. Build model (3 output neurons)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Must match num_classes
])

# 3. Compile with categorical crossentropy
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For multi-class
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    train_generator,
    epochs=5,
    steps_per_epoch=len(train_generator)
)

# 5. Save
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
    #IF RUNNING ON A DIFFERENT COMPUTER CHANGE THIS
model.save('/Users/anthonyjirano/Desktop/CSUF/CSUF Spring 2025/AI/Project Testing/skincare/models/skin_model_5_16.keras')