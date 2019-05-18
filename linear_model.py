import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error

customers = pd.read_csv('Ecommerce Customers.csv')

# print(customers.head())
# print(customers.info())
# print(customers.describe())
# print(customers.columns.values)

# plot data, draw a regression line and plot distribution for every axis
#inst = sns.JointGrid('Time on App', 'Yearly Amount Spent', customers)
#inst.plot(sns.regplot, sns.distplot)

#sns.jointplot('Time on App', 'Length of Membership', customers, kind='hex')

# plot every pair of data
#sns.pairplot(customers)


# plot data and draw a regression line
sns.lmplot('Length of Membership', 'Yearly Amount Spent', customers)

X = customers['Length of Membership']
Y = customers['Yearly Amount Spent']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, 
                                                    random_state = 101)

# reshape data to be a len x 1 matrix
x_train = np.array(x_train).reshape((-1, 1))
y_train = np.array(y_train).reshape((-1, 1))
x_test = np.array(x_test).reshape((-1, 1))
y_test = np.array(y_test).reshape((-1, 1))

# declare and train model
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model.fit(x_train, y_train)

# predict results
result = model.predict(x_test)

# measure accuracy using different metrics
MSE = mean_squared_error(y_test, result)
MAE = mean_absolute_error(y_test, result)
MSLE = mean_squared_log_error(y_test, result)
MEAE = median_absolute_error(y_test, result)
print("MSLE (mean squared log error): {}".format(MSLE))
print("MEAE (median absolute error): {}".format(MEAE))
print("MAE (mean absolute error): {}".format(MAE))
print("MSE (mean squared error): {}".format(MSE))

# plot correct and predicted points
fig = plt.figure()
axis = fig.add_subplot(111)
axis.scatter(x_test.reshape((1, -1)).tolist()[0], y_test.reshape((1, -1)).tolist()[0], label='correct', s=16)
axis.scatter(x_test.reshape((1, -1)).tolist()[0], result.reshape((1, -1)).tolist()[0], c='r', label='predicted', s=16)

plt.title('Correct vs predicted values')
plt.xlabel('Length of Membership')
plt.ylabel('Yearly Amount Spent')
fig.legend(loc='upper left')
plt.show()













