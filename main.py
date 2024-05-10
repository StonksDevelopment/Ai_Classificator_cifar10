import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np


def u_input(path):
    # преобразуем изображение в 32 на 32
    image = Image.open(path)
    resized = image.resize((32, 32))
    # создаём массив из изобрадения
    img_array = np.array(resized) / 255
    img_array = img_array.reshape((1, 32, 32, 3))
    predictions(img_array)

def predictions(img_array):
    # загружаем модель
    model = tf.keras.models.load_model("justmode.h5")

    # передаём в модель массив
    predictions = model.predict(img_array)
    chance = predictions[0]

    # Названия объектов
    classes = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]

    # Создаем гистограмму
    plt.bar(range(len(chance)), chance)

    # Добавляем названия по оси X
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Добавляем названия объектов и вероятности
    for i in range(len(chance)):
        plt.text(i, chance[i] + 0.05, f'{chance[i]: .2f}', ha='center', va='bottom')

    plt.show()

#вызываем функцию
print("Введите путь до изображения:")
u_path = str(input())
u_input(u_path)
