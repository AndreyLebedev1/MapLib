import time
from collections import Counter, defaultdict

import numpy as np
from drawMap import drawMap
from EM import EM as ElasticMap
from energyNorms import *
from minisom import MiniSom
from rect2DMap import rect2DMap
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


class SOMClassifier(MiniSom):
    def __init__(self, map_size: tuple = (10, 10), **kwargs):
        """
        Инициализация модели.

        :param map_size: Размер карты
        """
        self.map_size = map_size
        self.TrainedMap = None
        self.wieghts = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_time = None

    def load_data(self, X, y, test_size=0.2, random_state=42):
        """
        Загрузка и разделение данных на обучающую и тестовую выборки.

        :param X: Признаки
        :param y: Целевая переменная
        :param test_size: Доля тестовой выборки
        :param random_state: Сид для воспроизводимости
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train(self, num_iteration=500, **kwargs):
        """Обучение модели на обучающих данных."""
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Данные не загружены. Вызовите load_data() прежде чем обучать модель."
            )
        self.TrainedMap = MiniSom(
            x=self.map_size[0],
            y=self.map_size[1],
            input_len=self.X_train.shape[1],
            sigma=3,
            learning_rate=0.5,
            neighborhood_function="triangle",
            random_seed=10,
        )
        # super.__init__(x = self.map_size[0], y = self.map_size[1], input_len = self.X_train.shape[0], **kwargs)
        self.wieghts = self.TrainedMap.pca_weights_init(self.X_train)
        start_time = time.time()
        self.TrainedMap.train(self.X_train, num_iteration=num_iteration, verbose=False)
        end_time = time.time()

        self.train_time = round(end_time - start_time, 2)
        self.winmap = self.TrainedMap.labels_map(self.X_train, self.y_train)
        return self

    def predict(self, X=None):
        """
        Предсказание на тестовых данных или предоставленных данных.

        :param X: Признаки для предсказания (если None, используется тестовая выборка)
        :return: Предсказания модели
        """
        if X is None:
            X = self.X_test

        default_class = np.sum(list(self.winmap.values())).most_common()[0][0]
        result = []
        for d in X:
            win_position = self.TrainedMap.winner(d)
            if win_position in self.winmap:
                result.append(self.winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)

        return result

    def evaluate(self):
        """Оценка модели на тестовых данных."""
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Тестовые данные не загружены. Вызовите load_data() прежде чем оценивать модель."
            )

        print("=" * 100)
        print("TRAIN REPORT")
        print("TRAIN TIME, s: ", round(self.train_time, 2))
        print("=" * 100)

        start_time = time.time()
        prediction = self.predict(self.X_train)
        end_time = time.time()

        print(classification_report(prediction, self.y_train))
        print("Predict on TRAIN TIME, s: ", round(end_time - start_time, 2))
        print("=" * 100)

        print("=" * 100)
        print("TEST REPORT")
        print("=" * 100)

        start_time = time.time()
        prediction = self.predict(self.X_test)
        end_time = time.time()

        print(classification_report(prediction, self.y_test))
        print("Predict on TEST TIME, s: ", round(end_time - start_time, 2))
        print("=" * 100)


class ElasticMapClassifier:
    def __init__(self, map_size: tuple = (10, 10)):
        """
        Инициализация модели.

        :param map_size: Размер карты
        """
        self.map_size = map_size
        self.classes = np.zeros(map_size)

        self._map = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_time = None

    def load_data(self, X, y, test_size=0.2, random_state=42):
        """
        Загрузка и разделение данных на обучающую и тестовую выборки.

        :param X: Признаки
        :param y: Целевая переменная
        :param test_size: Доля тестовой выборки
        :param random_state: Сид для воспроизводимости
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train(self, plot=False):
        """Обучение модели на обучающих данных."""
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Данные не загружены. Вызовите load_data() прежде чем обучать модель."
            )

        if plot:
            plot = True
            plotFigure = True

        start_time = time.time()
        self._map = rect2DMap(self.map_size[0], self.map_size[1])
        self._map.init(self.X_train, "pci")
        colours = ["red", "blue"]

        ElasticMap(
            self._map,
            self.X_train,
            constStretching=0,
            constBending=0.1,
            func=L2,
            nInt=2,
            delta=0.75,
        )

        project_train = drawMap(
            self._map,
            self.X_train,
            classes=self.y_train,
            markColour=colours,
            newFigure=plot,
            plot=plot,
            lineWidth=0.5,
        )

        project_train = [(elem[0], elem[1]) for elem in project_train]
        winmap = defaultdict(list)

        for x, l in zip(project_train, self.y_train):
            winmap[x].append(l)

        for position in winmap:
            winmap[position] = Counter(winmap[position])

        self.winmap = winmap
        self.emap = {}
        for k in winmap.keys():
            self.emap.update({k: winmap[k].most_common(1)[0][0]})

        end_time = time.time()
        self.train_time = end_time - start_time

        return self

    def predict(self, X=None):
        """
        Предсказание на тестовых данных или предоставленных данных.

        :param X: Признаки для предсказания (если None, используется тестовая выборка)
        :return: Предсказания модели
        """
        colours = ["red", "blue"]

        if X is None:
            X = self.X_test

        project_test = drawMap(
            self._map,
            X,
            classes=np.ones((len(X), 1)),
            markColour=colours,
            newFigure=False,
            plot=False,
            lineWidth=0.5,
        )
        project_test = [(elem[0], elem[1]) for elem in project_test]
        res = []
        for k in project_test:
            res.append(self.emap[k])

        return res

    def evaluate(self):
        """Оценка модели на тестовых данных."""
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Тестовые данные не загружены. Вызовите load_data() прежде чем оценивать модель."
            )

        print("=" * 100)
        print("TRAIN REPORT")
        print("TRAIN TIME, s: ", round(self.train_time, 2))
        print("=" * 100)

        start_time = time.time()
        prediction = self.predict(self.X_train)
        end_time = time.time()

        print(classification_report(prediction, self.y_train))
        print("Predict on TRAIN TIME, s: ", round(end_time - start_time, 2))
        print("=" * 100)

        print("=" * 100)
        print("TEST REPORT")
        print("=" * 100)

        start_time = time.time()
        prediction = self.predict(self.X_test)
        end_time = time.time()

        print(classification_report(prediction, self.y_test))
        print("Predict on TEST TIME, s: ", round(end_time - start_time, 2))
        print("=" * 100)
