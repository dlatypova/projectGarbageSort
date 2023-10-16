#Импорт библиотеки глубокого обучения нейронной сети и ее модуля для работы с фото
import tensorflow as tf
from tensorflow.keras.preprocessing import image
#Импорт библиотеки ждя загрузки иодели обученной сети
from keras.models import load_model
#Импорт библиотеки графического инетерфеса и ее модулей
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
#Импорт библиотеки для работы с загруженным изображением
from PIL import Image, ImageTk
#Импорт библиотеки для работы с массивами изображений
import numpy as np

# Загрузка сохраненной модели
loaded_model = load_model('my_trained_model.h5')

# Функция для предсказания категории изображения
def classify_image():
    file_path = filedialog.askopenfilename()  # Запрос на выбор файла
    if file_path:
            #Чтение и обработка фото для отображения в графическом редакторе
            image_pil = Image.open(file_path)
            image_pil = image_pil.resize((512, 384))
            photo = ImageTk.PhotoImage(image_pil)

            # Отображаем изображение на интерфейсе
            label_image.config(image=photo)
            label_image.image = photo

            # Подготовка изображения для передачи в нашу обученную нейронную сеть для классификации
            test_image = image.load_img(file_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = tf.keras.applications.vgg19.preprocess_input(test_image)

            #Работа нейронной сети - выполнение предсказания категории
            predictions = loaded_model.predict(test_image)

            #Поиск индекса с наибольшей вероятностью и запись категории в перемнную
            class_index = np.argmax(predictions)
            class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
            predicted_class = class_names[class_index]

            result_label.config(text=f'Category: {predicted_class}', font='Georgia 30')
    #Обработка ошибки в случае, если пользователь не загрузил изображение
    else:
        messagebox.showwarning("Warning", "No image selected.")

app = tk.Tk()
app.title("Garbage Classification App")

app.geometry('800x600')
app['background'] = '#44C17A'

label_image = tk.Label(app)
label_image['background'] = '#44C17A'
label_image.pack()

result_label = tk.Label()
result_label['background'] = '#44C17A'
result_label.pack()

load_button = tk.Button(app, text="Load Image", command=classify_image, width = 50, height= 5, font='Georgia 15')
load_button.pack(pady=10)

app.mainloop()