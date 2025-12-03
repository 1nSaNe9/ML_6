import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Загрузка сохраненной модели
model = load_model('cnn_dog_cat_model.h5')


# Функция для предобработки изображения
def preprocess_image(image_path, img_size=(32, 32)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Функция для проверки на тестовых данных
def test_on_dataset():
    print("Этот функционал требует наличия тестовых данных.")
    print("Для проверки модели лучше используйте загрузку отдельных изображений.")


# Основная часть программы
def main():
    print("=" * 50)
    print("ПРОВЕРКА МОДЕЛИ ДЛЯ КЛАССИФИКАЦИИ СОБАК И КОШЕК")
    print("=" * 50)

    while True:
        print("\nВыберите действие:")
        print("1. Проверить на одном изображении")
        print("2. Выход")

        choice = input("Введите номер (1/2): ")

        if choice == '1':
            image_path = input("Введите путь к изображению: ")

            try:
                # Загрузка и предобработка
                img_array = preprocess_image(image_path)

                # Предсказание
                prediction = model.predict(img_array, verbose=0)
                dog_prob = prediction[0][0]
                cat_prob = prediction[0][1]

                print("\nРезультат предсказания:")
                print(f"Вероятность собаки: {dog_prob:.4f}")
                print(f"Вероятность кошки: {cat_prob:.4f}")

                if dog_prob > cat_prob:
                    print(f"Предсказание: СОБАКА (уверенность: {dog_prob:.2%})")
                else:
                    print(f"Предсказание: КОШКА (уверенность: {cat_prob:.2%})")

            except Exception as e:
                print(f"Ошибка: {e}")
                print("Убедитесь, что путь правильный и файл существует.")

        elif choice == '2':
            print("Выход из программы.")
            break

        else:
            print("Некорректный выбор. Попробуйте снова.")


if __name__ == '__main__':
    main()