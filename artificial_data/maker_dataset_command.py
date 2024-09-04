import random
import traceback
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.datasets import make_blobs

__author__ = "Carlos Castro"

class DatasetGenerator():
    """
    A class to generate an artificial PU (Positive Unlabeled) dataset.
    The intended use is for clustering algorithms,
    within the context of PUL (Positive Unlabeled Learning).
    It is possible to change the amount of data, number of groups,
    proportion of data in each group, proportion of positive values in a binary group,
    """
    
    def __init__(self, 
                n_points: int = 100,
                n_groups: int = 2,
                prop_groups: List[float] = [0.5],
                binary_groups: List[int] = [0, 1],
                gamma_pos: float = 0.05,
                include_outliers: bool = False,
                prop_outliers: float = 0.01,
                dimensions: int = 2,
                limits: tuple = ((0, 0), (500, 500)), # min and max
                similarity: List[float] = [0.5, 0.2],
                dissimilarity: List[float] = [0.7],
                distance_metric: str = 'euclidean',
                output_path: str = './dataset.csv'
                ) -> None:
        """
        Args:
            n_points: number of points.
            n_groups: number of groups.
            prop_groups: proportion of points in each group.
            binary_groups: group labels.
            gamma_pos: proportion of positive values in a binary group.
            include_outliers: if True, generates outliers.
            prop_outliers: proportion of outliers.
            dimensions: number of dimensions.
            limits: data limits.
            similarity: n_points/100 * similarity of the groups.
            dissimilarity: dissimilarity of the groups.
            distance_metric: distance metric.
            output_path: path to save the dataset        
        """
        try:
            self._n_points = n_points
            self._n_groups = n_groups            
            self._dimensions = dimensions            
            self._distance_metric = distance_metric
            self._gamma_pos = gamma_pos
            
            if len(binary_groups) != n_groups:                
                raise ValueError("The number of binary_groups values must be equal to the number of groups.")                
            self._binary_groups = binary_groups         
            if len(prop_groups) == 1:
                prop_groups = prop_groups * n_groups
            if len(prop_groups) != n_groups:    
                raise ValueError("The number of prop_groups values must be equal to the number of groups.")
            if sum(prop_groups) != 1:
                raise ValueError("The sum of prop_groups must be equal to 1.")
            self._prop_groups = prop_groups
            if len(similarity) == 1:
                similarity = similarity * n_groups
            if len(similarity) != n_groups:
                raise ValueError("The number of similarity values must be equal to the number of groups.")
            if 0 in similarity:
                raise ValueError("The similarity value must be greater than 0.")
            self._similarity = similarity           
            if len(dissimilarity) == 1:
                dissimilarity = dissimilarity * (n_groups - 1)
            if len(dissimilarity) != n_groups - 1:
                raise ValueError("The number of dissimilarity values must be equal to the number of groups - 1.")            
            self._dissimilarity = dissimilarity
            if prop_outliers <= 0 or prop_outliers >= 1:
                raise ValueError("The prop_outliers must be between 0 and 1.")
            self._prop_outliers = prop_outliers
            
            self._include_outliers = include_outliers
            if len(limits) != 2:
                raise ValueError("The limits must be a tuple with two elements.")
            if len(limits[0]) != dimensions or len(limits[1]) != dimensions:
                raise ValueError("The number of elements in the limits must be equal to the number of dimensions.")
            self._limits_min = limits[0]
            self._limits_max = limits[1]
            
            center_box = []
            for i in range(self._dimensions):
                center_box.append(self._limits_max[i] - self._limits_min[i])
            self._center_box = tuple(center_box)
                        
            self._output_path = output_path
            self._X, self._y, self._y_train = self.generate_dataset()
            
            self._data = pd.DataFrame(self._X, columns=[f"X{i}" for i in range(self._dimensions)])
            self._data['y'] = self._y
            self._data['y_train'] = self._y_train
            if self._output_path != "":
                self._data.to_csv(self._output_path, index=False)
                
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            raise e
    
    def plot(self, y: str = 'all', fix_scale: bool = False) -> None:
        if y == 'all':
            color_map = ['black' if yi == 0 else 'red' if yi == -1 else 'green' for yi in self._y]
        elif y == 'train':
            color_map = ['black' if yi == 0 else 'red' if yi == -1 else 'green' for yi in self._y_train]
        else:
            raise ValueError("Invalid value for y.")
        
       
        if self._dimensions == 2:
            plt.scatter(self._X[:, 0], self._X[:, 1], c=color_map)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Dataset 2d {y} class')
            if fix_scale:
                plt.gca().set_xlim([self._limits_min[0], self._limits_max[0]])
                plt.gca().set_ylim([self._limits_min[1], self._limits_max[1]])
                

        elif self._dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self._X[:, 0], self._X[:, 1], self._X[:, 2], c=color_map)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Dataset 3d {y} class')
            if fix_scale:
                plt.gca().set_xlim([self._limits_min[0], self._limits_max[0]])
                plt.gca().set_ylim([self._limits_min[1], self._limits_max[1]])
                plt.gca().set_zlim([self._limits_min[2], self._limits_max[2]])
        else:
            raise ValueError("The plot is only available for 2D and 3D data.")           
        plt.show()
        plt.close()
                        
    def get_arrays(self) -> tuple:
        return self._X, self._y, self._y_train
            
    def get_dataframe(self) -> pd.DataFrame:
        return self._data
    
    def random_point(self, point: tuple, distance: float) -> tuple:
        """
        Generates a random point from a given point and a distance.
        """
        while True:
            vector = np.random.randn(self._dimensions)
            vector /= np.linalg.norm(vector)        
            vector *= distance
            new_point = point + vector
            new_point = tuple(float(coord) for coord in new_point)
            in_limits = True
            
            for j in range(len(new_point)):
                coord = new_point[j]
                if coord < self._limits_min[j] or coord > self._limits_max[j]:
                    in_limits = False
            if in_limits:
                break
        new_point = tuple(float(coord) for coord in new_point)
        return new_point
    
    def distance(self, point1: tuple, point2: tuple) -> float:
        """
        Calculates the distance between two points in the specified metric.
        """
        def euclidean_distance(point1: tuple, point2: tuple) -> float:
            """
            Calculates the Euclidean distance between two points.
            """
            return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

        def manhattan_distance(point1: tuple, point2: tuple) -> float:
            """
            Calculates the Manhattan distance between two points.
            """
            return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

        def chebyshev_distance(point1: tuple, point2: tuple) -> float:
            """
            Calculates the Chebyshev distance between two points.
            """
            return max(abs(p1 - p2) for p1, p2 in zip(point1, point2))
    
        if self._distance_metric == 'euclidean':
            return euclidean_distance(point1, point2)
        elif self._distance_metric == 'manhattan':
            return manhattan_distance(point1, point2)
        elif self._distance_metric == 'chebyshev':
            return chebyshev_distance(point1, point2)
        else:
            raise ValueError(f"Invalid metric: {self._distance_metric}")
    
    def furthest_point(self, point: tuple) -> tuple:
        """
        Returns the furthest point from a given point.
        """
        def generate_combinations(limits_min, limits_max):
            """
            Generates all combinations of points between limits_min and limits_max.
            """
            combinations = list(itertools.product(*zip(limits_min, limits_max)))
            return combinations
        
        combinations = generate_combinations(self._limits_min, self._limits_max)
        
        furthest_point = point
                
        for xtrem in combinations:            
            bigger_distance = self.distance(point, furthest_point)
            actual_distance = self.distance(point, xtrem)            
            if actual_distance > bigger_distance:
                furthest_point = xtrem
        
        return furthest_point
            
    def convert_percentage(self, y: np.array, n: float, fm: int, to: int) -> np.array:
        """
        Function to convert a percentage of values from one class to another.
        Args:
            y: np.array with the labels.
            n: percentage of values to be converted.
            fm: value to be converted.
            to: value to which the values will be converted.
        Returns:
            np.array with the converted labels.
        """

        indices_y_from = np.where(y == fm)[0]
        num_convert = int(len(indices_y_from) * n)
        np.random.shuffle(indices_y_from)
        indices_convert = indices_y_from[:num_convert]
        y[indices_convert] = to
        return y
        
    def generate_centers(self) -> List[tuple]:
        """
        Generates the centers of the groups.
        """
        centers = []

        point = []            
        for dim in range(self._dimensions):               
            point.append(random.uniform(self._limits_min[dim], self._limits_max[dim]))
        point = tuple(point)
        centers.append(point)
        
        # Get the furthest point from the center
        furthest_point = self.furthest_point(point)
        dissimilarities = self._dissimilarity.copy()                       
        for i in range(1, self._n_groups):
            dissimilarity = random.choice(dissimilarities)
            dissimilarities.remove(dissimilarity)            
            distance = (self.distance(point, furthest_point) * dissimilarity) / 4           
            new_point = self.random_point(centers[i-1], distance)
            centers.append(new_point)                    
        
        return centers
    
    def generate_dataset(self) -> tuple:
        """
        Function to generate the dataset.        
        Returns:
            X: np.array with the data.
            y: np.array with the labels.
            y_train: np.array with the training labels.
        """
        
        # Generate the centers of the groups
        self._centers = self.generate_centers()
        
        if self._include_outliers:
            n_outliers = int(self._n_points * self._prop_outliers)
            X_outliers, _ = make_blobs(n_samples=n_outliers, centers=n_outliers, n_features=self._dimensions, cluster_std = n_outliers/2,
                  random_state=None, center_box=self._center_box)
            # Outliers are zeros
            y_outliers = np.zeros(n_outliers)
            X_aux = X_outliers
            y_aux = y_outliers
            y_train_aux = y_outliers           
        else:
            n_outliers = 0
            X_aux = np.array([]).reshape(0, self._dimensions)
            y_aux = np.array([])
            y_train_aux = np.array([])
            
        n_points_in_groups = self._n_points - n_outliers        
        prop_groups = self._prop_groups.copy()
        similarities = self._similarity.copy()
        binary_groups = self._binary_groups.copy()
                
        for center in self._centers:
            prop_group = random.choice(prop_groups)
            prop_groups.remove(prop_group)
            
            n_points = int(n_points_in_groups * prop_group)
            
            similarity = random.choice(similarities)
            similarities.remove(similarity)
            
            X, _ = make_blobs(n_samples=n_points, centers=1, n_features=self._dimensions, cluster_std = (n_points/(similarity*100)),
                    random_state=None, center_box=center)

            binary_group = random.choice(binary_groups)
            binary_groups.remove(binary_group)
            
            y = np.zeros(n_points) + binary_group            
            y_train = np.zeros(n_points)
            
            if binary_group == 1:                
                y_train = self.convert_percentage(y_train, self._gamma_pos, 0, 1)
            
            X_aux = np.concatenate((X_aux, X), axis=0)
            y_aux = np.concatenate((y_aux, y), axis=0)
            y_train_aux = np.concatenate((y_train_aux, y_train), axis=0)
        
        return X_aux, y_aux, y_train_aux

if __name__ == '__main__':
    obj = DatasetGenerator()
    X, y, y_train = obj.get_arrays()