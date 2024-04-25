import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
heart_data = pd.read_csv('heart.csv')

# print(heart_data)
# print(heart_data.isnull().sum())
# print(heart_data.keys())
#['Patient', 'Gender', 'Age', 'Cholesterol', 'Troponin', 'Smoker', 'BP','HeartAttack']
#        0       1           2       3           4           5          6    7

# print(heart_data['Gender'])
# print(heart_data['Smoker'])
# print(heart_data['HeartAttack'])

le_gender = LabelEncoder()
le_smoker = LabelEncoder()
le_heartattack = LabelEncoder()

heart_data['Gender'] = le_gender.fit_transform(heart_data['Gender'])
heart_data['Smoker'] = le_smoker.fit_transform(heart_data['Smoker'])
heart_data['HeartAttack'] = le_heartattack.fit_transform(heart_data['HeartAttack'])

# print(heart_data['Gender'])
# print(heart_data['Smoker'])
# print(heart_data['HeartAttack'])

x = heart_data.iloc[:,[1,2,3,4,5,6]].values
# print(x)
y = heart_data.iloc[:, -1].values
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = GaussianNB()
model.fit(x_train, y_train)

print('-----PREDICTION-----')
print(model.predict(x_test))
print('-----ACTUAL DATA-----')
print(x_test)

accuracy = model.score(x_test,y_test)
print('Accuracy:-  ',accuracy)

#['Patient', 'Gender', 'Age', 'Cholesterol', 'Troponin', 'Smoker', 'BP','HeartAttack']

Gender = input('[Male / Female]:- ')
age = int(input('AGE:- '))
Cholesterol = int(input('Cholesterol:- '))
Troponin = float(input('Troponin:- '))
BP = int(input('BLOOD PRESUURE:- '))
Smoker = input('Yes / No:- ')

gender = le_gender.transform([Gender])[0]
Smoker = le_smoker.transform([Smoker])[0]

prediction = model.predict([[gender,age,Cholesterol,Troponin,BP,Smoker]])

if prediction == 0:
    print('NO CHANCE OF HEART ATTACK')
else:
    print("LIKELY TO HAVE A HEART ATTACK")