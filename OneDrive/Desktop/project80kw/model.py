import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
data=pd.read_csv('data_robust1.csv')
#print(data.head())
data.drop(columns=['Unnamed: 0','Reactive_Power'],axis=1,inplace=True)
#print(data.head())
x=data.iloc[:,0:5]
y=data.iloc[:,-1]
#print(x)
#print(y)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(x)
X_scaled=pd.DataFrame(X_scaled,columns=x.columns)
#print(X_scaled)
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.20,random_state=123)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
model=RandomForestRegressor(max_depth=20,min_samples_leaf=2,min_samples_split=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#print(y_pred)
#print(model.score(X_train,y_train))
#print(model.score(X_test,y_test))
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

