# Проект: Сравнение изображений с использованием VAE

 Этот проект предоставляет решение для сравнения изображений с использованием вариационного автокодировщика (VAE) и различных метрик сходства. 
 Проект включает в себя графический интерфейс для загрузки изображений и выбора метрики, а также скрипты для инференса и поиска ближайших соседей по эмбеддингам.

## Структура проекта:
```
├── inference_loop.py            # Скрипт для инференса и получения эмбеддингов двух изображений
├── interface_prototype.py       # Прототип графического интерфейса на PyQt
├── model_VAE.py                # Классы моделей VAE для обучения и инференса
├── NNsearch.py                 # Модуль для поиска ближайших соседей по эмбеддингам
├── README.md                   # Описание проекта, установка, использование
```
# Описание файлов:

## 1. `inference_loop.py`
Скрипт для выполнения инференса модели, который на вход получает два изображения, прогоняет их через обученный VAE и выводит эмбеддинги этих изображений.

## 2. `interface_prototype.py`
Прототип графического интерфейса, построенный с использованием PyQt. Интерфейс позволяет пользователю загружать два изображения из локальной системы, выбирать метрику для их сравнения и передавать изображения для дальнейшего анализа.

## 3. `model_VAE.py`
Содержит определения моделей для вариационного автокодировщика (VAE). Здесь определены классы для кодировщика и декодировщика, а также сама модель VAE. Этот файл используется для обучения модели и генерации эмбеддингов.

## 4. `NNsearch.py`
Модуль для поиска ближайших соседей в латентном пространстве с использованием иерархического поиска. Этот модуль позволяет сравнивать эмбеддинги изображений, используя различные метрики сходства, такие как косинусное расстояние, евклидово расстояние и другие.


# Порядок запуска скриптов:

## 1. **Обучение модели VAE (если модель не обучена):**
Перед началом инференса необходимо обучить модель VAE. Это делается с использованием файла `model_VAE.py`. Обучение можно запустить в отдельном скрипте или модифицировать `model_VAE.py`, добавив тренировочный цикл.

## Запуск обучения модели VAE:
python model_VAE.py

## 2. **Запуск графического интерфейса:**
Запустите интерфейс для выбора изображений и метрики сравнения:

python interface_prototype.py

В этом интерфейсе вы сможете загрузить два изображения, выбрать метрику для их сравнения и получить эмбеддинги.

## 3. **Получение эмбеддингов и сравнение:**
В случае если вы хотите выполнить сравнение изображений через скрипт без GUI, используйте `inference_loop.py`, который позволяет получить эмбеддинги для двух изображений и выполнить их сравнение.
После получения эмбеддингов, можно использовать модуль `NNsearch.py` для поиска ближайших соседей и вычисления метрик сходства.

## Пример запуска инференса:
python inference_loop.py --image1 path_to_image1 --image2 path_to_image2

# Пример использования:

## 1. **Запустите графический интерфейс:**
Включите интерфейс, чтобы загрузить изображения и выбрать метрику для их сравнения:

python interface_prototype.py

## 2. **Запустите инференс напрямую (если не используете GUI):**
Для автоматического инференса и получения эмбеддингов изображений:

python inference_loop.py --image1 path_to_image1 --image2 path_to_image2

Получив эмбеддинги, вы можете использовать `NNsearch.py` для их сравнения.

