# %%
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from predictPy import Analisis_Predictivo
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# %%
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#import standardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#import PCA
from sklearn.decomposition import PCA


# %%
# Leer el df
df = pd.read_csv('raisin.csv')
df.head()

# %%
df.info()

# %%
#divide the df
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print (X.head())
print (y.head())

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state=0)

# %%
#crear el pipeline, aquí tambén podemos agregar una 
pipe = make_pipeline(StandardScaler(),PCA(n_components=2),SVC(kernel='rbf'))


# %% [markdown]
# Aquí se revisa todos los KERNELS

# %%
#create a dictionary with the hiperparameters to test
grid_parameters = [
                    {'svc__C': [1, 10], 'svc__gamma': [0.1]},
                    {'svc__C': [1, 10], 'svc__gamma': [0.1], 'svc__kernel': ['linear']},
                    {'svc__kernel': ['poly'], 'svc__degree': [2, 3, 4 ]},
                    {'svc__C': [1, 10], 'svc__gamma': [0.1], 'svc__kernel': ['sigmoid']},]
#create a gridsearch object
grid_search = GridSearchCV(pipe, grid_parameters) # fit grid seaarch

# %%

best_model = grid_search.fit(X_train, y_train) # fit the best model

# %%
#print the model settings and the best parameters
print("Best model: ", best_model.best_estimator_)
print("Best parameters: ", best_model.best_params_)
print("Best score: ", best_model.best_score_)


# %%
#predecir
y_pred = best_model.predict(X_test)


# %%
#print the classification report
print(classification_report(y_test, y_pred))
 

# %%
# print the confusion matrix
print(confusion_matrix(y_test, y_pred))


