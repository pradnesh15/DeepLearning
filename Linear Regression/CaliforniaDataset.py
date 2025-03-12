from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#load dataset
california=fetch_california_housing()
data=pd.DataFrame(california.data,columns=california.feature_names)
data['MedHouseValue']=california.target

#handling missing data
data=data.dropna()

#split data into features and target
X=data.drop(columns=['MedHouseValue'])
Y=data['MedHouseValue']

#split data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#normalize the state
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#train the model
lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)

#predict and evaluate
Y_pred_lin=lin_reg.predict(X_test)
mse_lin=mean_squared_error(Y_test,Y_pred_lin)
print(f'Linear Regression MSE={mse_lin}')