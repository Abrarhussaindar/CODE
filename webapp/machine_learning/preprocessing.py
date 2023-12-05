import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(test_size, path):
    df=pd.read_csv(r"C:\Users\abrar\Downloads\RePORT\CODE\webapp\machine_learning\dataset")


    df.drop(["Unnamed: 133"],axis=1,inplace=True)
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]

    pca=PCA(n_components = 18)
    pca.fit(x)
    x_pca=pca.transform(x)
    # x_pca.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=test_size, random_state=42)
    data = x_train,x_test,y_train,y_test
    return pca, data
