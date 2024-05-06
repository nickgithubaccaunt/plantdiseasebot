import os
from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import ContentType
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Токен вашего бота
API_TOKEN = ''

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Путь к модели для определения растений
plant_model_path = "model_weights.h5"
plant_model = load_model(plant_model_path)
plant_class_labels = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']

# Путь к модели для определения заболеваний
disease_model_path = 'model_weightsnew.h5'
disease_model = load_model(disease_model_path)
disease_class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew', 'Corn___Blight_Leaf',
                        'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
                        'Corn___Northern_Leaf_Blight', 'Grape___Black_Measles', 'Grape___Black_rot', 'Grape___healthy',
                        'Grape___Leaf_blight', 'Peach___Bacterial_spot', 'Peach___healthy',
                        'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight',
                        'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Strawberry___healthy',
                        'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                        'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# Путь к папке для сохранения изображений
image_folder = "D:/datasetforpractica/papkaval"

# Словарь с переводом заболеваний
disease_translation = {
    "Apple___Apple_scab": "Яблочная парша",
    "Apple___Black_rot": "Яблочная черная гниль",
    "Apple___Cedar_apple_rust": "Яблочная ржавчина",
    "Apple___healthy": "Яблоко здоровое",
    "Blueberry___healthy": "Черника здоровая",
    "Cherry___healthy": "Вишня здоровая",
    "Cherry___Powdery_mildew": "Вишня, пораженная мучнистой росой",
    "Corn___Blight_Leaf": "Кукуруза, пораженная фитофторозом листьев",
    "Corn___Cercospora_leaf_spot": "Кукуруза, пораженная церкоспорой листьев",
    "Corn___Common_rust": "Кукуруза, пораженная обыкновенной ржавчиной",
    "Corn___healthy": "Кукуруза здоровая",
    "Corn___Northern_Leaf_Blight": "Северная листовая гниль кукурузы",
    "Grape___Black_Measles": "Черная корь винограда",
    "Grape___Black_rot": "Черная гниль винограда",
    "Grape___healthy": "Виноград здоровый",
    "Grape___Leaf_blight": "Листовая гниль винограда",
    "Peach___Bacterial_spot": "Бактериальная пятнистость персика",
    "Peach___healthy": "Персик здоровый",
    "Pepper_bell___Bacterial_spot": "Жгучий перец бактериальная пятнистость",
    "Pepper_bell___healthy": "Жгучий перец здоровый",
    "Potato___Early_blight": "Картофель фитофтороз",
    "Potato___healthy": "Картофель здоровый",
    "Potato___Late_blight": "Картофель картофельная гниль",
    "Raspberry___healthy": "Малина здоровая",
    "Strawberry___healthy": "Клубника здоровая",
    "Strawberry___Leaf_scorch": "Клубника ожог листьев",
    "Tomato___Bacterial_spot": "Томаты бактериальная пятнистость",
    "Tomato___Early_blight": "Томаты ранний фитофтороз",
    "Tomato___healthy": "Томаты здоровые",
    "Tomato___Late_blight": "Томаты поздний фитофтороз",
    "Tomato___Leaf_Mold": "Томаты листовая плесень",
    "Tomato___Septoria_leaf_spot": "Томаты септориоз листьев",
    "Tomato___Spider_mites": "Томаты паутинные клещи",
    "Tomato___Target_Spot": "Томаты мишенеобразная пятнистость",
    "Tomato___Tomato_mosaic_virus": "Томаты вирус полосатой мозаики",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Томаты желтая курчавость листьев"
}


# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Нормализация значений пикселей от 0 до 1
    return img_array


# Функция для определения растения на изображении
def classify_plant(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = plant_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]

    return plant_class_labels[predicted_class_index], confidence


# Функция для определения заболевания растения
def detect_disease(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = disease_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]

    # Получаем класс заболевания
    disease_class = disease_class_labels[predicted_class_index]

    # Переводим класс заболевания, если перевод доступен
    translated_disease_class = disease_translation.get(disease_class, disease_class)

    return translated_disease_class, confidence


# Обработчик для изображений
@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    # Получаем объект фото
    photo = message.photo[-1]
    # Генерируем уникальное имя для сохранения фото
    file_id = photo.file_id
    file_path = os.path.join(image_folder, f'{file_id}.jpg')

    try:
        # Сохраняем фото
        await photo.download(file_path)

        # Проверяем, является ли изображение растением
        plant_class, plant_confidence = classify_plant(file_path)

        if plant_confidence > 0.5:
            # Если это растение, определяем его заболевание
            disease_class, disease_confidence = detect_disease(file_path)
            result = f"Заболевание: {disease_class} (Уверенность: {disease_confidence:.2f})"
        else:
            result = "Это не растение"

        # Отправляем результат пользователю
        await message.reply(result)

    except Exception as e:
        await message.reply(f"Произошла ошибка: {e}")

@dp.message_handler(commands=['help'])
async def handle_help(message: types.Message):
    help_text = "Для получения информации о растении, отправьте фото растения."
    await message.reply(help_text)


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply(
        "Бот разработан студентами МИИГАиК направления подготовки 'Прикладная информатика' в 2024 году в рамках учебного проекта.\n\n" +
        "Данный бот предназначен для распознавания болезней растений по фотографии листьев.\n\n" +
        "Участники проекта:\n" +
        "Роганов Никита.\n" +
        "Смыслов Алексей.\n" +
        "Лепёшкин Захар.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
