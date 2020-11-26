import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib

data=pd.read_csv('ToyotaCorolla.csv')
# print(data.corr())
y=data.Price
x=data.drop('Price',axis=1)
# x=x.drop('Age_08_04',axis=1)
# x=x.drop('KM',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# regr=linear_model.LinearRegression()
# regr.fit(x_train,y_train)
# joblib.dump(regr,'model_joblib')
model=joblib.load('model_joblib')
predi=model.predict(x_test)
print(y_test,predi)
print('Coefficient of determination: %.2f'%r2_score(y_test, predi))


