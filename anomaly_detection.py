import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.cluster import DBSCAN
from preprocessing import DataProcessor
import matplotlib.pyplot as plt


class OutlierDetector:

    """
    Determines whether a given sample is considered outlier
    """

    def __init__(self, df, sample):
        self.df = df
        self.sample = sample

        return


    def compute_gower(self):
        """
        compute the gower distance 

        :param dataset: <str> the name of the dataset
        :return: <numpy array> returns the distance array as a numpy array
        """

        individual_variable_distances = []
        sum_total_weights = 0

        for i in range(self.df.shape[1]):
            feature = self.df.iloc[:,[i]]
            if feature.dtypes.values == np.object:
                feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
                weight = 0.8
                sum_total_weights += weight
                
            else:
                feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / max(np.ptp(feature.values),1)
                weight = 1
                sum_total_weights += weight

            individual_variable_distances.append(feature_dist * weight)
         
        return np.array(individual_variable_distances).sum(0) / sum_total_weights

        
    def find_clusters(self, dist_matrix, eps = 0.25, min_samples = 40):
        """
        find the clusters through DBSCAN

        :param dist_matrix: <numpy array> the distance matrix 
        :param eps: <float> the local radius for expanding clusters 
        :param min_samoles: <int> the minimum size of each cluster formed         
        :return: <numpy array> returns the cluster for each datapoint
        """
        clusters = (DBSCAN(eps = eps, 
                            min_samples = min_samples,
                            metric='precomputed')
                    .fit(dist_matrix)
                    .labels_)    

        return clusters


    def outlier_ckeck(self):
        """
        check whether the given sample is an outlier

        :param dist_matrix: <numpy array> the distance matrix        
        :return: <boolean> returns a boolean flag of whether the sample is outlier
        """
        dist_matrix = self.compute_gower()
        clusters = self.find_clusters(dist_matrix)

        flag = clusters[self.sample] == -1

        return flag

    
    def numeric_dist(self):
        """
        calculate and plot the distances of the numerical variables

        :param dataset: <str> the name of the dataset
        :return: <numpy array> returns the distance array as a numpy array
        """
        individual_variable_distances = []

        for i in range(self.df.shape[1]):
            feature = self.df.iloc[:,[i]]
            if feature.dtypes.values == np.object:
                continue     
            else:
                feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / max(np.ptp(feature.values),1)
                individual_variable_distances.append(feature_dist)

        num_dist_matrix = np.array(individual_variable_distances).sum(0) 

        plt.figure()
        plt.subplot(111)
        plt.boxplot(num_dist_matrix.mean(axis=0))
        return 


if __name__ == '__main__':

    processor = DataProcessor("data/german/german.data")
    df = processor.load_data()    

    detector = OutlierDetector(df,1)
    flag = detector.outlier_ckeck()
    print(flag)
