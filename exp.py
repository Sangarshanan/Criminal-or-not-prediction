import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('criminal_train.csv')

train = data.drop('Criminal',axis =1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, data['Criminal'], test_size=0.30,random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)