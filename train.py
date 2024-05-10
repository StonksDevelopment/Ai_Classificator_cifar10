import tensorflow as tf


def train():
    # загружаем датасет из тензерфлоу
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

    # делим на 255 т.к rgb
    x_train = x_train / 255
    x_val = x_val / 255

    # создаём матрицы
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)

    # создаём модель
    model = tf.keras.Sequential([
        # создаём слои
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(1000, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")

    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=4, epochs=10, validation_data=(x_val, y_val))

    model.save('justmode.h5')

train()
