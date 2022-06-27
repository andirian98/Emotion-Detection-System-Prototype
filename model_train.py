from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
import keras
import os
from keras import regularizers

def train_app(folder,filename,size_batch,size_epoch,rate_learn):
        num_classes = 7
        img_rows, img_cols = 48, 48
        batch_size = size_batch

        train_dir = folder + '/train'
        val_dir = folder + '/test'
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(img_rows,img_cols),
                batch_size=size_batch,
                color_mode="grayscale",
                class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(48,48),
                batch_size=size_batch,
                color_mode="grayscale",
                class_mode='categorical')

        emotion_model = keras.models.Sequential([
                # 1st Block
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # 2nd Block
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # 3rd Block
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # 4th Block
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # 5th Block
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                    input_shape=(img_rows, img_cols, 1), activation='elu'),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
                # Fully Connected Layer (FCL)
                # 5th Block
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='elu', kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(0.001)),
                # 6th Block
                keras.layers.Dense(64, activation='elu', kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(0.001)),
                # 7th Block
                keras.layers.Dense(num_classes, kernel_initializer='he_normal', activation='softmax'),
        ])
        emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=rate_learn, decay=1e-6),metrics=['accuracy'])

        totalDir = 0
        totalFiles = 0
        totalDir1 = 0
        totalFiles1 = 0
        for base, dirs, files in os.walk(train_dir):
                for directories in dirs:
                        totalDir += 1
                for Files in files:
                        totalFiles += 1
        print("Total Files: ", totalFiles)

        for base1, dirs1, files1 in os.walk(val_dir):
                for directories in dirs1:
                        totalDir1 += 1
                for Files in files1:
                        totalFiles1 += 1
        print("Total Files: ", totalFiles1)
        nb_train_samples = totalFiles
        nb_validation_samples = totalFiles1
        epochs = size_epoch

        emotion_model_info = emotion_model.fit(
                train_generator,
                steps_per_epoch=nb_train_samples // size_batch,
                epochs=epochs,
                batch_size=size_batch,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples // size_batch)

        emotion_model.save_weights(filename)

        emotion_model.save(filename)

