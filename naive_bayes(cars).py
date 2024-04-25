import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

car_data = pd.read_csv('cars.csv')
# print(car_data['Car'])
# print(car_data['Color'])

# Encode categorical variables separately
le_car = LabelEncoder()
le_color = LabelEncoder()

car_data['Car'] = le_car.fit_transform(car_data['Car'])
car_data['Color'] = le_color.fit_transform(car_data['Color'])
# print(car_data['Car'])
# print(car_data['Color'])

x = car_data.iloc[:, :-1].values
y = car_data.iloc[:, -1].values
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = GaussianNB()
model.fit(x_train, y_train)

print('-----PREDICTION-----')
print(model.predict(x_test))
print('-----ACTUAL DATA-----')
print(x_test)

accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

'''
# Make predictions on the test set
predictions = model.predict(x_test)

# Calculate accuracy manually
correct_predictions = (predictions == y_test).sum()
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions
print("Manually calculated accuracy:", accuracy)
'''

car = input('Ferrari/Ford/Toyota/Benz: ')
color = input('Cyan/Red/Black: ')
HP = int(input('Horse Power: '))

car = le_car.transform([car])[0]
color = le_color.transform([color])[0]

prediction = model.predict([[car, color, HP]])

if prediction == 0:
    print("The user will not purchase")
else:
    print("The user is likely to purchase")

