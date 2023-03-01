#START
#import Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)

#imprt Dataset
dataset = pd.read_csv("C:\\Users\\S P Mithun\\diabetes.csv")


#Removing Null values
cols = ["Glucose","BloodPressure","Insulin","SkinThickness","BMI"]
to_see = []
for col in cols:
    unex_values = dataset[dataset[col] == 0]
    to_see.append(unex_values)
to_see

dataset[cols] = dataset[cols].replace(0,np.nan)

for col in cols:
    dataset[col] = dataset[col].fillna(dataset[col].median())

#Data split into train and test of ratio 80:20
dataset['Outcome'].value_counts()
dataset.groupby('Outcome').mean()

X = dataset.drop(columns = 'Outcome', axis=1)
Y = dataset['Outcome']

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

#Bagging Ensemble learning or Bootstrap Aggregation
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=2)
cart = DecisionTreeClassifier()
num_trees = 10
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print('Accuracy: ', results.mean())

model.fit(X_train, Y_train)


#Prediction
input_data = (1,189,60,23,846,30.1,0.398,59)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Not Diabetic')
else:
  print('Is Diabetic')
  

#Save model
filename = 'model1.pkl'
pickle.dump(model, open(filename, 'wb'))  
  

loaded_model = pickle.load(open('model1.pkl', 'rb'))


input_data = (2,197,70,45,543,30.5,0.158,53)

input_data_as_numpy_array = np.asarray(input_data)


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Not Diabetic')
else:
  print('Is Diabetic')

#END