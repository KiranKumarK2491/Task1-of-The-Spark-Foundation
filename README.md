# Task1-of-The-Spark-Foundation
Data Science &amp; Business Analytics Intern (#GRIPApril21)
Submitted By : KIRAN KUMAR K
OBJECT : Prediction using Supervised ML
Task 1 : To predict the percentage of an student based on the no. of study hours.
What will be predicted score if a student studies for 9.25 hrs/ day?
Data can be found at http://bit.ly/w-data
Solution : Using Linear Regression
Steps 1 : Import required libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
Step 2 : Import data
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data has been imported successfully")
data.head(5)
Data has been imported successfully
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
Step 3 : Summarizing and plotting the data
data.describe()
Hours	Scores
count	25.000000	25.000000
mean	5.012000	51.480000
std	2.525094	25.286887
min	1.100000	17.000000
25%	2.700000	30.000000
50%	4.800000	47.000000
75%	7.400000	75.000000
max	9.200000	95.000000
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()

Step 4 : Divide the data into independent and dependent variables & Split the data into training data and testing data
X =  data.iloc[:, :-1].values
Y =  data.iloc[:, 1].values
X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
Step 5 : Fitting the linear regression model and plotting it
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print("Model fitted")
Model fitted
line = regressor.coef_*X+regressor.intercept_
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()

Step 6 : Predicting the scores of test set
print(X_test) 
y_pred = regressor.predict(X_test)
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]
 [3.8]
 [1.9]
 [7.8]]
df = pd.DataFrame({'Actual' : Y_test, 'Predicted' : y_pred})
df
Actual	Predicted
0	20	17.053665
1	27	33.694229
2	69	74.806209
3	30	26.842232
4	62	60.123359
5	35	39.567369
6	24	20.969092
7	86	78.721636
Step 7 : Model evaluation (finding root mean squared error)
print('Root Mean Squared Error :', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
Root Mean Squared Error : 4.792191274636316
Step 8 : Finding score of a student who has studied for 9.25 hrs/day
hour = [9.25]
ans = regressor.predict([hour])
print("Score = {}".format(round(ans[0],3)))
Score = 92.915
Conclusion
The task for predicting the scores based on No. of studying hours was completed using the ML trained data
The results that we found are as follows :
Root Mean Squared Error : 4.792191274636316
No of Hours studied = 9.25¶
Predicted Score = 92.915
Thank you :)
