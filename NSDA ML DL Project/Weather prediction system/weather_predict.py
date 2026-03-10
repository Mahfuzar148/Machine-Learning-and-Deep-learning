#import necessary libraries for developing this model
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 





df = pd.read_csv('weatherAUS.csv')

df.head()

df.info()
df.shape
df.describe()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

df.columns

df.select_dtypes(include='object').columns

le = LabelEncoder()

col_encode = ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm',
       'RainToday', 'RainTomorrow']

for col in col_encode:
  df[col] = le.fit_transform(df[col])

df.head()


#feature selection 

X = df.drop(['RainToday'], axis=1)
y = df['RainToday']

y.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

#Data_scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model1 = DecisionTreeClassifier()
model1.fit(X_train, y_train)

model2 = RandomForestClassifier()
model2.fit(X_train, y_train)

logistic = model.score(X_test, y_test)
decision = model1.score(X_test, y_test)
random = model2.score(X_test, y_test)

logistic, decision, random

models = ['logistic', 'decision_tree', 'random_forest']
accuracy = [logistic, decision, random]

plt.bar(models, accuracy)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()


#train accuracy 
y_pred_train = model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
print('Training accuracy : ',train_acc)


#Testing accuracy 
y_pred_test = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print('Testing accuracy : ',test_acc)


#overfit and undeerfit checking 
'''
train high , test low ---> Overfitting 
train low , test low---> Underfitting 
train high , test high ---> Generalization
'''

if train_acc > 0.90 and test_acc<0.70:
  print('Overfitting')
elif train_acc < 0.70 and test_acc<0.70:
  print('Underfitting')
else:
  print('Model is balanced/generalized')



joblib.dump(model, 'weather_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')



