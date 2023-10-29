# hacks-ai-tbo

## Запуска через Python

1. Установите зависимости из `requirements.txt`
2. Запустите `main.py`

## Запуск через bat

1. Скачайте демо по [ссылке](https://disk.yandex.ru/d/4cfUDH0YX9Y-RQ)
2. Запустите start.bat


## Структура проекта
1. `analysis/` - jupyter notebooks с исследованиями мультиспектральных фото
   1. `channels.ipynb` - исследование каналов
   2. `detection.ipynb` - исследование bbox на каналах
2. `parser/` - модуль с парсерами в различные форматы
   1. `coco.py` - исходный датасет в coco формат
   2. `to_csv.py` - выход программы из ТЗ в csv для отправки
   3. `yolo_detection.py` - исходный датасет в yolo формат
3. `test/`
   1. `frames.py` - подсчёт метрик на изначальном датасете
4. `tracking/` - YOLO Tracking при помощи SORT
   1. `tracking.py` - преобразование данного сэмлпа в количество объектов по кадрам и в целом
   2. `tracking_yolo_v8x.py` - cv2 графики с bbox от yolo и изначальной аннотацией
5. `conver.py` - скрипт для преобразованя данного датасета в yolo датасет
6. `create_coco.py` - создать coco датасет из исходного
7. `main.py` - запуск программы, ввод названия видео, запуск YOLO
8. `train.ipynb` - jupyter notebook для тренировки модели

## Где взять модель YOLO?
В разделе [Github Releases](https://github.com/mihett05/hacks-ai-tbo/releases/download/model/yolo_v8l.pt) можно скачать файл `yolo_v8l.pt`. Его необходимо положить в корневую директорию

## Куда положить видео для анализа?
В папку `train_dataset_dataset` положить папку видео, затем укажите это название при запуске программы
