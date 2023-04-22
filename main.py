import pandas as pd
import numpy as np
import sys
!{sys.executable} -m pip install mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_data_set = load_iris()

#print (iris_data_set.keys())
print (iris_data_set['feature_names'])

xTrain, xTest, yTrain, yTest = train_test_split(iris_data_set['data'], iris_data_set['target'], random_state=0)

iris_dataframe = pd.DataFrame(xTrain, columns=iris_data_set.feature_names)
display(iris_dataframe)

