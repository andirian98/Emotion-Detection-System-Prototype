import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras import regularizers

def main_vgg(folder,filename,size_batch,size_epoch,rate_learn):
    num_classes = 7
    img_rows, img_cols = 48, 48
    batch_size = size_batch

    train_data_dir = folder + '/train/'
    validation_data_dir = folder + '/test/'

    #performing data augmentation
    train_datagenerate = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.3, zoom_range=0.3,
                                            width_shift_range=0.4,
                                            height_shift_range=0.4, horizontal_flip=True, fill_mode='nearest')

    test_validation_datagenerate = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagenerate.flow_from_directory(train_data_dir, color_mode='grayscale',
                                                             target_size=(img_rows, img_cols),
                                                             batch_size=batch_size, class_mode='categorical',shuffle=True)

    test_validation_generator = test_validation_datagenerate.flow_from_directory(validation_data_dir,
                                                                                 color_mode='grayscale',
                                                                                 target_size=(img_rows, img_cols),
                                                                                 batch_size=batch_size,
                                                                                 class_mode='categorical',shuffle=False)

    model = keras.models.Sequential([
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
    print(model.summary())

    checkpoint = ModelCheckpoint(filename,
                                 monitor='val_accuracy',
                                 mode='auto',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1)

    earlystop = EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=size_epoch,
                              verbose=1,
                              mode='auto',
                              restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                  factor=0.2,
                                  patience=size_epoch,
                                  verbose=1,
                                  min_delta=0.0001)

    csv_logger = CSVLogger("vgg16_result.csv", append=True)

    callbacks = [earlystop, checkpoint, reduce_lr, csv_logger]

    model.compile(loss='categorical_crossentropy',
                  loss_weights=0.3, optimizer=SGD(learning_rate=rate_learn, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    totalDir = 0
    totalFiles = 0
    totalDir1 = 0
    totalFiles1 = 0
    for base, dirs, files in os.walk(train_data_dir):
        for directories in dirs:
            totalDir+=1
        for Files in files:
            totalFiles+=1
    print("Total Files: ",totalFiles)

    for base1, dirs1, files1 in os.walk(validation_data_dir):
        for directories in dirs1:
            totalDir1+=1
        for Files in files1:
            totalFiles1+=1
    print("Total Files: ",totalFiles1)
    nb_train_samples = totalFiles
    nb_validation_samples = totalFiles1
    epochs = size_epoch

    history = model.fit(
                    train_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=epochs,
                    batch_size = batch_size,
                    callbacks=callbacks,
                    validation_data=test_validation_generator,
                    validation_steps=nb_validation_samples//batch_size)

    print('Testing dataset')
    Y_pred = model.predict(test_validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print(confusion_matrix(test_validation_generator.classes, y_pred))
    print('-------------------------------------------------------------------------------------')
    class_labels = list(test_validation_generator.class_indices.keys())
    print('Classification Report')
    target_names = class_labels
    print('Target names')
    print(classification_report(test_validation_generator.classes, y_pred, target_names=target_names, zero_division=1))
    print('Labels')
    print(classification_report(test_validation_generator.classes, y_pred, labels=np.unique(y_pred), zero_division=1))

    print('Training dataset')
    Z_pred = model.predict(train_generator)
    z_pred = np.argmax(Z_pred, axis=1)
    print(confusion_matrix(train_generator.classes, z_pred))
    print('-------------------------------------------------------------------------------------')
    class_labels = list(train_generator.class_indices.keys())
    print('Classification Report')
    target_names = class_labels
    print('Target names')
    print(classification_report(train_generator.classes, z_pred, target_names=target_names, zero_division=1))
    print('Labels')
    print(classification_report(train_generator.classes, z_pred, labels=np.unique(z_pred), zero_division=1))


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(epochs)

    fig1, axs1 = plt.subplots()
    axs1.plot(epochs, acc, 'g', label='Training accuracy')
    axs1.plot(epochs, val_acc, 'b', label='validation accuracy')
    axs1.set_title('Training and Validation accuracy')
    axs1.set_xlabel('Epochs')
    axs1.set_ylabel('Accuracy')
    axs1.legend()

    fig2, axs2 = plt.subplots()
    axs2.plot(epochs, loss, 'g', label='Training loss')
    axs2.plot(epochs, val_loss, 'b', label='validation loss')
    axs2.set_title('Training and Validation loss')
    axs2.set_xlabel('Epochs')
    axs2.set_ylabel('Loss')
    axs2.legend()

    plt.tight_layout()
    plt.show()
