import sys
sys.path.append(r'./utils')
from utils.baselineCorrectorUtils import *
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class Res:
    CRLS: np.ndarray = None
    CRTS: np.ndarray = None
    WRLS: np.ndarray = None
    WRTS: np.ndarray = None
    centre: np.ndarray = None
    project: np.ndarray = None
    centroid1: np.ndarray = None
    centroids: np.ndarray = None
    FD1: np.ndarray = None
    FD: np.ndarray = None


class BaselineCorrector:
    def __init__(self):
        self.corrector_state = None

    def fit(self, CRLS, CRTS, WRLS, WRTS, space='reduced', whitening=False, setPC=3, numPC=150, numClust=15):
        result = Res()

        # Инициализация
        result.CRLS = CRLS
        result.CRTS = CRTS
        result.WRLS = WRLS
        result.WRTS = WRTS

        # Расчет центра в зависимости от setPC
        match setPC:
            case 1:
                all = np.concatenate((CRLS, WRLS), axis=0)
                centre = np.mean(all, axis=0)
            case 2:
                centre = np.mean(CRLS, axis=0)
            case 3:
                centre = np.mean(WRLS, axis=0)
            case _:
                raise ValueError("Invalid value for setPC. Valid options are: 1 (CRLS+WRLS), 2 (CRLS), or 3 (WRLS).")

        result.centre = centre
        result.space = space

        # Проверка на корректность значения space
        if space == "original":
            # Центрирование данных
            CRLS = CRLS - result.centre
            WRLS = WRLS - result.centre
            
            if whitening:
                match setPC:
                    case 1:
                        both = np.concatenate((CRLS, WRLS), axis=0)
                        project = np.std(both, axis=0)
                    case 2:
                        project = np.std(CRLS, axis=0)
                    case 3:
                        project = np.std(WRLS, axis=0)
            else:
                project = np.ones(result.centre.shape)

            result.project = project

            # Нормализация данных
            CRLSR = np.divide(CRLS, project, where=project!=0, out=np.zeros_like(CRLS))
            WRLSR = np.divide(WRLS, project, where=project!=0, out=np.zeros_like(WRLS))

        elif space == "reduced":
            # PCA для уменьшения размерности
            pca = PCA(n_components=numPC, whiten=whitening)
            match setPC:
                case 1:
                    both = np.concatenate((CRLS, WRLS), axis=0)
                    pca.fit(both)
                case 2:
                    pca.fit(CRLS)
                case 3:
                    pca.fit(WRLS)
            CRLSR = pca.transform(CRLS)
            WRLSR = pca.transform(WRLS)
            result.project = pca.components_.T
        else:
            raise ValueError("Invalid value for space. Valid options are: 'original' or 'reduced'.")

        # Кластеризация
        result.centroid1 = np.zeros((1, WRLSR.shape[1]))
        kmeans = KMeans(n_clusters=numClust, random_state=0, n_init="auto").fit(WRLSR)
        result.centroids = kmeans.cluster_centers_

        labels = kmeans.labels_
        for i in range(numClust):
            if not (labels == i).sum():
                labels = [x-1 if x > i else x for x in labels]
                result.centroids = np.delete(result.centroids, i, axis=0)
        numClust = result.centroids.shape[0]

        # Расчет дискриминантных векторов
        covCRLSR = np.cov(CRLSR.T)
        meanCRLSR = np.mean(CRLSR, axis=0)

        result.FD1 = np.linalg.solve(covCRLSR + np.cov(WRLSR.T), meanCRLSR - np.mean(WRLSR, axis=0))
        result.FD1 /= np.sqrt(np.sum(result.FD1 ** 2))

        result.FD = np.zeros((numPC, numClust))
        for k in range(numClust):
            tmp = WRLSR[np.where(labels == k)]
            if tmp.shape[0] > 1:
                cov1 = np.cov(tmp.T)
            else:
                cov1 = np.zeros((WRLSR.shape[1], WRLSR.shape[1]))
            result.FD[:, k] = np.linalg.solve(covCRLSR + cov1, meanCRLSR - np.mean(tmp, axis=0))

        result.FD /= np.sqrt(np.sum(result.FD ** 2, axis=0))
        self.corrector_state = result
        return None


    def correct(self, X):
        projection_crls_train = splitToClusters(
            data = (self.corrector_state.CRLS[:10000] - self.corrector_state.centre) @ self.corrector_state.project,
            centroids = self.corrector_state.centroids,
            FD = self.corrector_state.FD
        )

        projection_wrls_train = splitToClusters(
            data = (self.corrector_state.WRLS[:10000] - self.corrector_state.centre) @ self.corrector_state.project,
            centroids = self.corrector_state.centroids,
            FD = self.corrector_state.FD
        )

        projection_crls_test = splitToClusters(
            data = (X - self.corrector_state.centre) @ self.corrector_state.project,
            centroids = self.corrector_state.centroids,
            FD = self.corrector_state.FD
        )

        projection_wrls_test = splitToClusters(
            data = (X - self.corrector_state.centre) @ self.corrector_state.project,
            centroids = self.corrector_state.centroids,
            FD = self.corrector_state.FD
        )

        error_prediction = []

        class Namespace: pass
        res = Namespace()

        nBins = 100

        nClust = len(projection_crls_train)

        CRLProjClust_train = projection_crls_train
        WRLProjClust_train = projection_wrls_train

        CRLProjClust_test = projection_crls_test
        WRLProjClust_test = projection_wrls_test

        res.clustersTrainingStat = np.zeros((nClust, 6))
        res.clustersTestStat = np.zeros((nClust, 9))
        for k in range(nClust):
            res.clustersTrainingStat[k] = oneClusterGraph(CRLProjClust_train[k], WRLProjClust_train[k], round(nBins / 10.),
                    "Distribution of training set for cluster %03d" % k)
            
            res.clustersTestStat[k] = oneClusterGraph(CRLProjClust_test[k], WRLProjClust_test[k], round(nBins / 10.),
                    "Distribution of test set for cluster %03d" % k, res.clustersTrainingStat[k, 2])

            # print(res.clustersTestStat[k, 2], CRLProjClust_test[k])

            for elem in CRLProjClust_test[k]:
                if elem > res.clustersTestStat[k, 2]:
                    error_prediction.append('CR')
                else:
                    error_prediction.append('WR')

        return [error_prediction]
    
