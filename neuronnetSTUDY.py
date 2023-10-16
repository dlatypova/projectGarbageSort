# Импорт библиотеки для работы с массивами
import numpy as np

# Импорт библеотеки для задач глубокого обучения
import tensorflow as tf

# Импорт разных компонентов библиотек для создания, компиляции и обучения модели, а также для обработки изображений и создания генераторов данных
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Пути для папок с обучающими и тестовыми данными
train_path = 'C:\\Users\\Home\\Downloads\\Garbage\\train_path_date'
valid_path = 'C:\\Users\\Home\\Downloads\\Garbage\\valid_path_date'


# Создает генератор данных для тренировочного набора, используя модель vgg19 и настраивая параметры данных
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.1).flow_from_directory(
    directory=train_path, target_size=(224,224), classes=['cardboard', 'glass', 'metal',
                                                         'paper', 'plastic', 'trash'], batch_size=16, subset='training')


# Создание генератора данных для тестового набора
valid_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    validation_split=0.1).flow_from_directory(
    directory=valid_path, target_size=(224,224), classes=['cardboard', 'glass', 'metal',
                                                         'paper', 'plastic', 'trash'], batch_size=16, subset='validation')
IMG_SIZE = 224                        #размер изображений
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)   #форма изображений

# Импортируем базу свертки модели VGG16 с предварительно заданными весами
base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                        include_top=False,
                                        weights='imagenet')

#Готовая модель слоев, которую можно редактировать
model = Sequential()

# Замораживаем веса
base_model.trainable=False

# Добавление сверточной базы VGG19 для инициализации последовательной модели
model.add(base_model)

# Добавляется слой для усреднения признаков по всему изображению
model.add(GlobalAveragePooling2D())

#Добавление полносвязного слоя с 512 скрытыми единицами(высокоуровненвое извлечение признаков)
model.add(Dense(units=512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#Добавление полносвязного слоя с 128 скрытыми единицами
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Добавление полносвязного слоя с 6 скрытыми единицами(выходами)
model.add(Dense(units=6, activation='softmax'))
#Визуализация модели
model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#Обучение модели на 20 эпохах
model_details = model.fit(x=train_batches, validation_data=valid_batches, epochs=20, verbose=2)

#Извлечение данных мониторинга
loss = model_details.history['loss']
validation_loss = model_details.history['val_loss']
accuracy = model_details.history['accuracy']
validation_accuracy = model_details.history['val_accuracy']

#Разморозка весов для дообучения
base_model.trainable=True

#Обучение сети на 4 эпохах и извлечение данных
model_details = model.fit(x=train_batches, validation_data=valid_batches, epochs=4, verbose=2)
loss.extend(model_details.history['loss'])
validation_loss.extend(model_details.history['val_loss'])
accuracy.extend(model_details.history['accuracy'])
validation_accuracy.extend(model_details.history['val_accuracy'])