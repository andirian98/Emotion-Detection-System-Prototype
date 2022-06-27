import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
#from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score, recall_score, precision_recall_fscore_support
import numpy as np

def main_googlenet(folder,filename,size_batch,size_epoch,rate_learn):
    num_classes = 7
    img_rows, img_cols = 48, 48
    batch_size = size_batch

    train_data_dir = folder + '/train'
    validation_data_dir = folder + '/test'

    train_datagenerate = ImageDataGenerator(rescale=1./255, rotation_range=30, shear_range=0.3, zoom_range=0.3,
                                            width_shift_range=0.4,
                                            height_shift_range=0.4, horizontal_flip=True, fill_mode='nearest')

    test_validation_datagenerate = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagenerate.flow_from_directory(train_data_dir, color_mode='grayscale',
                                                             target_size=(img_rows, img_cols),
                                                             batch_size=batch_size, class_mode='categorical',
                                                             shuffle=True)

    test_validation_generator = test_validation_datagenerate.flow_from_directory(validation_data_dir,
                                                                                 color_mode='grayscale',
                                                                                 target_size=(img_rows, img_cols),
                                                                                 batch_size=batch_size,
                                                                                 class_mode='categorical',
                                                                                 shuffle=False)

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)


    def inception_module(x,
                         filters_1x1,
                         filters_3x3_reduce,
                         filters_3x3,
                         filters_5x5_reduce,
                         filters_5x5,
                         filters_pool_proj,
                         name=None):
        conv_1x1 = keras.layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='elu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

        conv_3x3 = keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='elu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_3x3 = keras.layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='elu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

        conv_5x5 = keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='elu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_5x5 = keras.layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='elu',
                                       kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

        pool_proj = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = keras.layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='elu',
                                        kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

        output = keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

        return output

    input_layer = keras.layers.Input(shape=(img_rows, img_cols, 1))

    x = keras.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='elu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = keras.layers.MaxPool2D((2, 2), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = keras.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='elu', name='conv_2a_3x3/1')(x)
    x = keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='elu', name='conv_2b_3x3/1')(x)
    x = keras.layers.MaxPool2D((2, 2), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = keras.layers.MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')


    x1 = keras.layers.AveragePooling2D((2, 2), strides=3)(x)
    x1 = keras.layers.Conv2D(128, (1, 1), padding='same', activation='elu')(x1)
    x1 = keras.layers.Flatten()(x1)
    x1 = keras.layers.Dense(1024, activation='elu',kernel_regularizer=regularizers.l2(0.001))(x1)
    #x1 = keras.layers.Dropout(0.7)(x1)
    x1 = keras.layers.Dense(num_classes, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')

    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')


    x2 = keras.layers.AveragePooling2D((2, 2), strides=3)(x)
    x2 = keras.layers.Conv2D(128, (1, 1), padding='same', activation='elu')(x2)
    x2 = keras.layers.Flatten()(x2)
    x2 = keras.layers.Dense(1024, activation='elu',kernel_regularizer=regularizers.l2(0.001))(x2)
    #x2 = keras.layers.Dropout(0.7)(x2)
    x2 = keras.layers.Dense(num_classes, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')

    x = keras.layers.MaxPool2D((2, 2), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    x = keras.layers.GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    #x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    model.summary()

    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    model.summary()

    checkpoint = ModelCheckpoint(filename,
                                 monitor='output_accuracy',
                                 mode='auto',
                                 save_best_only=True,
                                 save_weights_only=False,
                                 verbose=1)

    earlystop = EarlyStopping(monitor='output_accuracy',
                              min_delta=0,
                              patience=size_epoch,
                              verbose=1,
                              mode='auto',
                              restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='output_accuracy',
                                  factor=0.2,
                                  patience=size_epoch,
                                  verbose=1,
                                  min_delta=0.0001)

    csv_logger = CSVLogger("GoogleNet_result.csv", append=True)

    callbacks = [earlystop, checkpoint, reduce_lr, csv_logger]

    # use loss corrrectly
    # categorical_crossentropy -  if you have one-hot encoded your target in order to have 2D shap
    # sparse_categorical_crossentropy - if you have 1D integer encoded target
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[1, 0.3, 0.3], optimizer=SGD(lr=rate_learn, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    totalDir = 0
    totalFiles = 0
    totalDir1 = 0
    totalFiles1 = 0
    for base, dirs, files in os.walk(train_data_dir):
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1
    print("Total Files: ", totalFiles)

    for base1, dirs1, files1 in os.walk(validation_data_dir):
        for directories in dirs1:
            totalDir1 += 1
        for Files in files1:
            totalFiles1 += 1
    print("Total Files: ", totalFiles1)
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

    #loss
    loss = history.history['loss']
    output_loss = history.history['output_loss']
    auxilliary_output_1_loss = history.history['auxilliary_output_1_loss']
    auxilliary_output_2_loss = history.history['auxilliary_output_2_loss']
    val_loss = history.history['val_loss']
    val_output_loss = history.history['val_output_loss']
    val_auxilliary_output_1_loss = history.history['val_auxilliary_output_1_loss']
    val_auxilliary_output_2_loss = history.history['val_auxilliary_output_2_loss']

    #accuracy
    output_acc = history.history['output_accuracy']
    auxilliary_output_1_accuracy = history.history['auxilliary_output_1_accuracy']
    auxilliary_output_2_accuracy = history.history['auxilliary_output_2_accuracy']
    val_output_acc = history.history['val_output_accuracy']
    val_auxilliary_output_1_accuracy = history.history['val_auxilliary_output_1_accuracy']
    val_auxilliary_output_2_accuracy = history.history['val_auxilliary_output_2_accuracy']

    epochs = range(epochs)

    fig1, axs1 = plt.subplots()
    axs1.plot(epochs, output_acc, 'g', label='Output accuracy')
    axs1.plot(epochs, auxilliary_output_1_accuracy, 'r', label='Auxilliary Output 1 accuracy')
    axs1.plot(epochs, auxilliary_output_2_accuracy, 'm', label='Auxilliary Output 2 accuracy')
    axs1.plot(epochs, val_output_acc, 'b', label='Validation accuracy')
    axs1.plot(epochs, val_auxilliary_output_1_accuracy, 'c', label='Validation Auxilliary Output 1 accuracy')
    axs1.plot(epochs, val_auxilliary_output_2_accuracy, 'y', label='Validation Auxilliary Output 2 accuracy')
    axs1.set_title('Output, Auxilliary Output, Validation & Validation Auxilliary Output accuracy')
    axs1.set_xlabel('Epochs')
    axs1.set_ylabel('Accuracy')
    axs1.legend()

    fig2, axs2 = plt.subplots()
    axs2.plot(epochs, loss, 'k', label='Loss')
    axs2.plot(epochs, output_loss, 'g', label='Output Loss')
    axs2.plot(epochs, auxilliary_output_1_loss, 'r', label='Auxilliary Output 1 loss')
    axs2.plot(epochs, auxilliary_output_2_loss, 'm', label='Auxilliary Output 2 loss')
    axs2.plot(epochs, val_loss, 'b', label='Validation loss')
    axs2.plot(epochs, val_output_loss, 'c', label='Validation Output loss')
    axs2.plot(epochs, val_auxilliary_output_1_loss, 'c', label='Validation Auxilliary Output 1 loss')
    axs2.plot(epochs, val_auxilliary_output_2_loss, 'y', label='Validation Auxilliary Output 2 loss')
    axs2.set_title('Output, Auxilliary Output, Validation & Validation Auxilliary Output loss')
    axs2.set_xlabel('Epochs')
    axs2.set_ylabel('Loss')
    axs2.legend()

    plt.tight_layout()
    plt.show()



