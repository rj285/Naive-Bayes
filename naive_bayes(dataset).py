import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data_set = pd.read_csv("dataset.csv")
# print(data_set)

x = data_set.iloc[:,[1,2,3]].values
print(x)
y = data_set.iloc[:,-1].values
# print(y)

le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])
print(x)

train_test_split(x,y,test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# niave byers model

model = GaussianNB()
model.fit(x_train, y_train)

print('-----PREDICTION-----')
print(model.predict(x_test))
print('-----ACTUAL DATA-----')
print(x_test)

gender = input("[MALE / FEMALE]:- ")
age = int(input("age:- "))
salary = int(input('salary:- '))

gender = le.fit_transform([gender])[0]
prediction = model.predict([[gender, age, salary]])

if prediction == 0:
    print("The user will not purchase")
else:
    print("The user is likely to purchase")