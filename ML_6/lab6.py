import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import kagglehub

# --------- Загрузка датасета ---------
print("Загрузка датасета...")
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print("Путь к датасету:", path)

# --------- Исследуем структуру датасета ---------
print("\nИсследуем структуру датасета...")
for root, dirs, files in os.walk(path, topdown=True):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if len(files) > 0:
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            print(f'{indent}  изображений: {len(image_files)}')


# --------- Загрузка изображений ---------
def load_images(base_path, img_size=(32, 32), max_per_class=2000):
    images = []
    labels = []

    # Ищем папки с изображениями
    for root, dirs, files in os.walk(base_path):
        # Определяем класс по имени папки/файла
        folder_name = os.path.basename(root).lower()

        # Определяем, собака или кошка
        class_idx = None
        if 'dog' in folder_name:
            class_idx = 0
        elif 'cat' in folder_name:
            class_idx = 1

        if class_idx is not None:
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = 0

            for filename in image_files:
                if count >= max_per_class:
                    break

                try:
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(img_size)
                    images.append(np.array(img))
                    labels.append(class_idx)
                    count += 1
                except:
                    continue

    # Если не нашли по папкам, ищем по всем файлам
    if len(images) == 0:
        print("Не найдены структурированные папки, ищем все изображения...")
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Определяем класс по имени файла
                        file_lower = filename.lower()
                        class_idx = None

                        if 'dog' in file_lower:
                            class_idx = 0
                        elif 'cat' in file_lower:
                            class_idx = 1

                        if class_idx is not None and len(images) < max_per_class * 2:
                            img_path = os.path.join(root, filename)
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize(img_size)
                            images.append(np.array(img))
                            labels.append(class_idx)
                    except:
                        continue

    return np.array(images), np.array(labels)


# Загружаем данные
train_data, train_labels = load_images(path)
print(f"\nЗагружено изображений: {len(train_data)}")

if len(train_data) == 0:
    print("ОШИБКА: Не удалось загрузить изображения!")
    print("Проверьте структуру датасета. Возможно, нужно изменить логику поиска изображений.")
    exit()

# --------- Предобработка данных ---------
train_data = train_data.astype('float32') / 255.0
train_labels = to_categorical(train_labels, num_classes=2)

# --------- Создание модели ---------
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --------- Обучение модели ---------
print("\nНачинаем обучение...")
model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.1)

# --------- Сохранение модели ---------
model.save('cnn_dog_cat_model.h5')
print("\nМодель сохранена в файл 'cnn_dog_cat_model.h5'.")