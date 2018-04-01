import os
import json
import models
import numpy as np
from keras.utils import np_utils
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.datasets import cifar10, cifar100, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, SGD
from Eve import Eve
import matplotlib.pyplot as plt

def set_lr_schedule(n):
    #epoch_divide_lr = [150,225]
    #if np.isin(n, epoch_divide_lr):
    #        K.set_value(nn.optimizer.lr, K.get_value(nn.optimizer.lr) / 10.0)
    lr = 1e-3
    if n > 180:
        lr = lr * 0.5e-3
    elif n > 160:
        lr = lr * 1e-3
    elif n > 120:
        lr = lr * 1e-2
    elif n > 80:
        lr = lr * 1e-1
    print('Learning rate: ', lr)
    #K.set_value(nn.optimizer.lr, lr)
    return lr

def train(model_name, **kwargs):
    """
    Train model

    args: model_name (str, keras model name)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    #TODO:
        # pass a list of GPUs + CUDA_VISIBLE_DEVICES so parallel scripts possible
        # Add Squeeze-and-excite Nets and WideResNets

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs["nb_epoch"]
    dataset = kwargs["dataset"]
    optimizer = kwargs["optimizer"]
    experiment_name = kwargs["experiment_name"]
    n_gpus = kwargs["n_gpus"]
    
    subtract_pixel_mean = True
    
    if dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    if dataset == "cifar100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
        X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.
    
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

    img_dim = X_train.shape[-3:]
    nb_classes = len(np.unique(y_train))

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)


    # Compile model.
    if optimizer == "SGD":
        opt = SGD(lr=1E-2, decay=1E-4, momentum=0.9, nesterov=True)
    if optimizer == "Adam":
        opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-4)
    if optimizer == "Eve":
        opt = Eve(lr=1E-4, decay=1E-4, beta_1=0.9, beta_2=0.999, beta_3=0.999, small_k=0.1, big_K=10, epsilon=1e-08)
        
   # MultiGPU
    if n_gpus > 1:
        #import tensorflow as tf
        #with tf.device('/cpu:0'):
        singlemodel = models.load(model_name, img_dim, nb_classes)
        model = multi_gpu_model(singlemodel, n_gpus)
    else:
        singlemodel = models.load(model_name, img_dim, nb_classes)
        model = singlemodel
        
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    singlemodel.summary()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)
    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    
    lr_scheduler = LearningRateScheduler(set_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
    
    board = TensorBoard(log_dir='./logs')

    callbacks = [lr_reducer, lr_scheduler, board]

    #for e in range(nb_epoch):
        #print(e)
        #set_lr_schedule(model, e)
        #print(K.get_value(model.optimizer.lr))
          
        #loss = model.fit(X_train, Y_train,
        #                 batch_size=batch_size * n_gpus,
        #                 validation_data=(X_test, Y_test),
        #                 epochs=1)
        
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size * n_gpus),
                         validation_data=(X_test, Y_test),
                         epochs=nb_epoch, verbose=1, workers=4, callbacks=callbacks)
    
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, 1 + len(history_dict['acc']) )
    
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
    '''
    train_losses.append(loss.history["loss"])
    val_losses.append(loss.history["val_loss"])
    train_accs.append(loss.history["acc"])
    val_accs.append(loss.history["val_acc"])

    # Save experimental log
    d_log = {}
    d_log["experiment_name"] = experiment_name
    d_log["img_dim"] = img_dim
    d_log["batch_size"] = batch_size
    d_log["nb_epoch"] = nb_epoch
    d_log["train_losses"] = train_losses
    d_log["val_losses"] = val_losses
    d_log["train_accs"] = train_accs
    d_log["val_accs"] = val_accs
    d_log["optimizer"] = opt.get_config()
    # Add model architecture
    # BUG - SEEMINGLY INCOMPATIBLE WITH MULTIGPU
    json_string = json.loads(singlemodel.to_json())
    for key in json_string.keys():
        d_log[key] = json_string[key]
    json_file = os.path.join("log", '%s_%s_%s.json' % (dataset, singlemodel.name, experiment_name))
    with open(json_file, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)
    '''