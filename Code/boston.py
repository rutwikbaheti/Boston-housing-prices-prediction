from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
bostan_dataset=load_boston()
print(bostan_dataset.keys())
x_train,x_test,y_train,y_test=train_test_split(bostan_dataset['data'],bostan_dataset['target'],random_state=0)
regression=LinearRegression()
regression.fit(x_train,y_train)
pred=regression.predict(x_test)
print(regression.score(x_test,y_test))
print(metrics.mean_squared_error(y_test,pred))
print(metrics.mean_absolute_error(y_test,pred))
df_result=pd.DataFrame({'Actual':y_test,'Predicted':pred})
print(df_result)