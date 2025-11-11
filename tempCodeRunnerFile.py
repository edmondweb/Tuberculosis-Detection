# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set up directories (modify with your paths)
train_dir = 'C:/Users/user/Desktop/Projects/Final project/Tuberculosis Detection/dataset/Test'  # Path to training data
val_dir = 'C:/Users/user/Desktop/Projects/Final project/Tuberculosis Detection/dataset/Validation'  # Path to validation data
test_dir = 'C:/Users/user/Desktop/Projects/Final project/Tuberculosis Detection/dataset/Test'  # Path to test data

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# Image data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation data

test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for test data

# Load the training, validation, and testing data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),  # Resize images to 512x512
    batch_size=16,
    class_mode='binary',  # Binary classification (TB vs Normal)
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode='binary',
    shuffle=False  # No shuffling for validation data
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode='binary',
    shuffle=False  # No shuffling for test data
)

# Build the model using ResNet50 as the base (pre-trained)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

# Freeze the layers of the base model
base_model.trainable = True  # Set to True to fine-tune the model

# Freeze the first 10 layers of the base model
for layer in base_model.layers[:140]:
    layer.trainable = False

# Build a custom model on top of the pre-trained base
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Reduces dimensionality
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification (TB vs Normal)
])

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # You can increase this based on your data
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stopping]
)

# Save the model after training
model.save('tb_detection_model.h5')  # Saves the model to a file

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.2f}")

# Optionally: Make predictions on the test set
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)

# Convert predictions to binary labels (0 or 1)
predictions_binary = (predictions > 0.5).astype("int32")

# Print some example predictions
for i in range(5):
    print(f"Prediction: {predictions_binary[i]}, True label: {test_generator.labels[i]}")
 