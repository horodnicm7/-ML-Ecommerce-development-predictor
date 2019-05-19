import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error
from sklearn.metrics import explained_variance_score


# suppress warning thrown by seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

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
#sns.lmplot('Length of Membership', 'Yearly Amount Spent', customers)

X = customers[['Length of Membership', 'Time on App', 'Time on Website', 'Avg. Session Length']]
Y = customers['Yearly Amount Spent']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, 
                                                    random_state = 101)

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
print("RMSE (sqrt of mean squared error): {}".format(np.sqrt(MSE)))
print("MSE (mean squared error): {}\n".format(MSE))

# a coefficient means that for every unit your data is increased with, the 
# output will be increased with the coefficient
coefficients = pd.DataFrame(model.coef_, x_train.columns, columns=['Coefficient'])
print(coefficients)

# it should be as high as possible for a good model
print("\nVariance score: {}".format(explained_variance_score(y_test, result)))

# plot difference between correct and predicted values
sns.distplot((y_test - result), bins=50)
plt.show() # force the above plot to be shown

# plot data. The output should form a linear equation as much as possible
plt.scatter(y_test, result, s=16)

plt.title('Correct vs predicted values')
plt.xlabel('Test output')
plt.ylabel('Predicted output')
plt.show()
