# train_emotion_recognition.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import your modules (update import paths as needed)
from models.cnn import create_complete_model
from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

# Ensure TensorFlow is using GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters
batch_size = 32
num_epochs = 30  # Will likely be stopped earlier by early stopping
input_shape = (64, 64, 1)
validation_split = 0.2
verbose = 1
num_classes = 7
patience = 50
base_path = './models/'

# Create output directory if it doesn't exist
if not os.path.exists(base_path):
    os.makedirs(base_path)

# Data generator for data augmentation
data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

# Create and compile model
model = create_complete_model(input_shape, num_classes)
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.summary()

# Process each dataset
datasets = ['fer2013']
for dataset_name in datasets:
    print(f'Training dataset: {dataset_name}')

    # Callbacks
    log_file_path = os.path.join(base_path, f'{dataset_name}_emotion_training.log')
    csv_logger = CSVLogger(log_file_path, append=False)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=1,
        restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=int(patience/4),
        verbose=1,
        min_lr=0.00001)
    
    # model_path = os.path.join(base_path, f'{dataset_name}_mini_XCEPTION')
    # model_checkpoint = ModelCheckpoint(
    #     filepath=model_path + '.{epoch:02d}-{val_accuracy:.2f}.keras',
    #     monitor='val_loss',
    #     verbose=1,
    #     save_best_only=True)
    
    callbacks = [csv_logger, early_stopping, reduce_lr]

    # Loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    
    # Split data
    train_data, val_data = split_data(faces, emotions, validation_split)
    train_faces, train_emotions = train_data
    val_faces, val_emotions = val_data
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_faces) // batch_size
    
    # Train the model
    history = model.fit(
        data_generator.flow(train_faces, train_emotions, batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=(val_faces, val_emotions))
    
    # Save the final model
    # final_model_path = os.path.join(base_path, f'{dataset_name}_mini_XCEPTION_final.keras')
    # model.save(final_model_path)
    # print(f"Model saved to {final_model_path}")
    val_acc = history.history.get('val_accuracy', [None])[-1]
    if val_acc is not None:
        val_acc_str = f"{val_acc:.4f}"
    else:
        val_acc_str = "NA"

    # Tạo tên file có độ chính xác
    final_model_path = os.path.join(base_path, f'{dataset_name}_ilabcnn_final_acc_{val_acc_str}.keras')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")


print("Training completed!")