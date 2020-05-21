"""
@File : End2endNet.py
@Author: Dong Wang
@Date : 2020/2/28
"""
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def End2endNet(pretrained_weights=None, input_size=(240, 320, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(80, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(conv1)

    conv2 = Conv2D(160, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(conv2)

    conv3 = Conv2D(320, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(conv3)

    conv4 = Conv2D(640, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    drop4 = Dropout(0.25)(conv4)

    conv5 = Conv2D(320, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=[2, 2])(conv5)
    Flat5 = Flatten()(pool5)
    Dense6 = Dense(160, activation='relu', kernel_initializer='he_normal')(Flat5)
    Dense7 = Dense(6, activation='softmax')(Dense6)

    model = Model(inputs=inputs, outputs=Dense7)

    # decay_rate = 1e-6
    # opt = RMSprop(learning_rate=learning_rate, decay=decay_rate)

    learning_rate = 3 * 1e-4
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def train(model, x, y, epochs, batch_size, validation_split, save_path='./saved_models/'):
    print('[Model] Training Start')
    print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
    save_fname = save_path + 'End2endWeights-best.hdf5'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True),
    ]
    history = model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        validation_split=validation_split,
        verbose=1
    )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.show()
    plt.savefig('./imgs/loss_fig.png')

    print('[Info] loss and accuracy figure generated in ./imgs/loss_fig.png ')
