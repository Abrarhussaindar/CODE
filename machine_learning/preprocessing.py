import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("C:\\Users\\YMTS0618\\Downloads\\Training.csv\\Training.csv")


df.drop(["Unnamed: 133"],axis=1,inplace=True)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

pca=PCA(n_components = 18)
pca.fit(x)
x_pca=pca.transform(x)
x_pca.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

