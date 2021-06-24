import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import matplotlib.pyplot as plt


breast_cancer=load_breast_cancer()
data= breast_cancer.data
features= breast_cancer.feature_names
df= pd.DataFrame(data, columns=features)
print(features)

df_small= df.iloc[:,:6]
covariance_mat=df_small.cov()
correlation_mat= df_small.corr()

sns.heatmap(covariance_mat,annot=True)
sns.heatmap(correlation_mat, annot=True)

plt.show()


data_BL=pd.read_csv("Mkt_Prices.csv", header=0)

Data_BL2=data_BL.cov()

Data_BL2

