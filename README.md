# Репозиторий для построения и тестирования классификаторов на базе карт Коханена и гибких карт.
В файле maps.py содержатся обертки над алгоритмами для построения сущности классификаторов с возможным сохранением его состояния и тестирования.

# ElasticMapClassifier

`ElasticMapClassifier` — это класс для классификации данных с использованием эластичных карт. Он предоставляет методы для загрузки данных, обучения модели и оценки ее производительности.

## Установка

Убедитесь, что у вас установлены необходимые библиотеки:

```bash
pip install numpy scikit-learn
```
## Импорт класса
```python
from maps import ElasticMapClassifier
```

## Инициализация модели
```python
model = ElasticMapClassifier(map_size=(10, 10))
map_size: размер карты в формате (ширина, высота).
```

## Загрузка данных
```python
model.load_data(X, y, test_size=0.2, random_state=42)
X: признаки (фичи) ваших данных.
y: целевая переменная.
test_size: доля тестовой выборки (по умолчанию 0.2).
random_state: сид для воспроизводимости (по умолчанию 42).
```

## Обучение модели
```python
model.train(plot=False)
plot: если True, будет построен график.
```
## Предсказание
```python
predictions = model.predict(X=None)  # Если X=None, используются тестовые данные
X: признаки для предсказания. Если не указано, будут использованы тестовые данные.
```
## Оценка модели
```python
model.evaluate()
```

## Пример
```python
import numpy as np
from sklearn.datasets import load_iris
```

# Загрузка данных
```python
data = load_iris()
X, y = data.data, data.target
```

# Инициализация и обучение модели
```python
model = ElasticMapClassifier(map_size=(10, 10))
model.load_data(X, y)
model.train(plot=True)
```

# Оценка модели
```python
model.evaluate()
```

# Методы
```python
load_data(X, y, test_size=0.2, random_state=42): Загружает и разделяет данные на обучающую и тестовую выборки.
train(plot=False): Обучает модель на обучающих данных.
predict(X=None): Предсказывает метки классов для тестовых или предоставленных данных.
evaluate(): Оценивает производительность модели на тестовых данных.
```

# Примечания
Перед обучением модели необходимо вызвать метод load_data().
При вызове метода evaluate() необходимо убедиться, что данные были загружены.

# SOMClassifier

`SOMClassifier` — это класс для классификации данных с использованием самоорганизующихся карт (Self-Organizing Maps, SOM). Он наследуется от `MiniSom` и предоставляет удобные методы для загрузки данных, обучения модели и оценки ее производительности.

## Установка
Убедитесь, что у вас установлены следующие библиотеки:

```bash
pip install numpy scikit-learn minisom
```

## Загрузка данных
```python
data = load_iris()
X, y = data.data, data.target
```
## Инициализация и обучение модели
```python
model = SOMClassifier(map_size=(10, 10))
model.load_data(X, y)
model.train(num_iteration=500)
```
## Оценка модели
```python
model.evaluate()
```
## Методы
```python
load_data(X, y, test_size=0.2, random_state=42): Загружает и разделяет данные на обучающую и тестовую выборки.
train(num_iteration=500): Обучает модель на обучающих данных.
predict(X=None): Предсказывает метки классов для тестовых или предоставленных данных.
evaluate(): Оценивает производительность модели на тестовых данных.
```
## Примечания
Перед обучением модели необходимо вызвать метод load_data().
При вызове метода evaluate() необходимо убедиться, что данные были загружены.

# Запуск скрипта из консоли

Для запуска скрипта в вашем проекте выполните следующую команду в консоли:

```bash
{путь к репозиторию в консоли}/Maplib % python {путь к .py скрипту} "{путь к бинарнику с данными}"
```

# Запуск примеров
```bash
/home/user/Maplib % python /home/user/Maplib/example_runner_EM_classifier.py  "./bin/example_data.pickle"
```
```
/home/user/Maplib % python /home/user/Maplib/example_runner_SOM_classifier.py  "./bin/example_data.pickle"
```
