import pandas as pd

dataset = pd.read_csv('SalaryData.csv')

y = dataset['Salary']
x = dataset['YearsExperience'].values.reshape(30,1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x,y)

n = int(input("Enter the year of Experience : "))

result = model.predict([[n]])

print("Predicted Salary for {} years of Experience is : {}".format(n,result))
