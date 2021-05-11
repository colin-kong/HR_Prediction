```python
# For ignoring future warning messages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

%matplotlib inline
sns.set()
```

Importing data into pandas dataframe from csv file


```python
df = pd.read_csv("HR_dataset.csv")
```

Doing data exploration


```python
df.sample(10)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>names</th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>exp_in_company</th>
      <th>work_accident</th>
      <th>left_company</th>
      <th>promotion_last_5years</th>
      <th>role</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1339</th>
      <td>Michael Quan</td>
      <td>0.89</td>
      <td>0.96</td>
      <td>5</td>
      <td>221</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
    </tr>
    <tr>
      <th>10738</th>
      <td>Anna Farruggio</td>
      <td>0.68</td>
      <td>0.84</td>
      <td>3</td>
      <td>270</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>support</td>
      <td>high</td>
    </tr>
    <tr>
      <th>4838</th>
      <td>Joseph Eastman</td>
      <td>0.46</td>
      <td>0.38</td>
      <td>6</td>
      <td>165</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>accounting</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4107</th>
      <td>James Mast</td>
      <td>0.55</td>
      <td>0.80</td>
      <td>3</td>
      <td>254</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IT</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>4540</th>
      <td>Michael Stevenson</td>
      <td>0.54</td>
      <td>0.85</td>
      <td>4</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>technical</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>8649</th>
      <td>Elizabeth Walker</td>
      <td>0.58</td>
      <td>0.88</td>
      <td>5</td>
      <td>178</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>14276</th>
      <td>Donnell Maybury</td>
      <td>0.81</td>
      <td>0.70</td>
      <td>6</td>
      <td>161</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>IT</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>4652</th>
      <td>Manuel Warner</td>
      <td>0.45</td>
      <td>0.58</td>
      <td>3</td>
      <td>200</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>RandD</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>13029</th>
      <td>Greg Palmer</td>
      <td>0.63</td>
      <td>0.49</td>
      <td>4</td>
      <td>151</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>technical</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>529</th>
      <td>Wilson Linkous</td>
      <td>0.39</td>
      <td>0.57</td>
      <td>2</td>
      <td>145</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>accounting</td>
      <td>medium</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>exp_in_company</th>
      <th>work_accident</th>
      <th>left_company</th>
      <th>promotion_last_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15000.000000</td>
      <td>15000.000000</td>
      <td>15000.000000</td>
      <td>15000.000000</td>
      <td>15000.000000</td>
      <td>15000.000000</td>
      <td>15000.000000</td>
      <td>15000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.612847</td>
      <td>0.716115</td>
      <td>3.803133</td>
      <td>201.052400</td>
      <td>3.498333</td>
      <td>0.144600</td>
      <td>0.238133</td>
      <td>0.021267</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.248628</td>
      <td>0.171171</td>
      <td>1.232590</td>
      <td>49.942074</td>
      <td>1.460139</td>
      <td>0.351709</td>
      <td>0.425955</td>
      <td>0.144277</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.090000</td>
      <td>0.360000</td>
      <td>2.000000</td>
      <td>96.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.440000</td>
      <td>0.560000</td>
      <td>3.000000</td>
      <td>156.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.640000</td>
      <td>0.720000</td>
      <td>4.000000</td>
      <td>200.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.820000</td>
      <td>0.870000</td>
      <td>5.000000</td>
      <td>245.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>310.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.countplot(df['role'])
```


    
![png](output_6_0.png)
    



```python
ax = sns.countplot(x="role", hue= "left_company", data=df)
```


    
![png](output_7_0.png)
    



```python
corr = df.corr()
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(corr, annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fde97f28810>




    
![png](output_8_1.png)
    


**Observation**


1.  Maximum number of years in the company is 10 years and 75% of the employees are working 4 years or below in this company.
2.   Sales department has highest number of employees and highest turn over
3.   Attrition  rate of this company is 23%

* the dataset does not require data cleaning


For this prediction I am going to use KNN model.


```python
inputs = ['satisfaction_level','number_project', 'average_monthly_hours']
X = df[inputs]
y = df['left_company']
```

Splitting data for training(70%) and testing(30%)


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

Train KNeighborsRegressor and show prediction accuracy


```python
from sklearn.neighbors import KNeighborsRegressor


knn_regressor1 = KNeighborsRegressor(n_neighbors=3)
knn_regressor1.fit(X_train, y_train)

knn_regressor2 = KNeighborsRegressor(n_neighbors=2)
knn_regressor2.fit(X_train, y_train)

knn_regressor3 = KNeighborsRegressor(n_neighbors=4)
knn_regressor3.fit(X_train, y_train)

print('KNeighborsRegressor1 :', knn_regressor1.score(X_test, y_test))
print('KNeighborsRegressor2 :', knn_regressor2.score(X_test, y_test))
print('KNeighborsRegressor3 :', knn_regressor3.score(X_test, y_test))
```

    KNeighborsRegressor1 : 0.6386993718865064
    KNeighborsRegressor2 : 0.6597831384015594
    KNeighborsRegressor3 : 0.6272691276803118
    

Train KNeighborsClassifier and show prediction accuracy


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn_classifier1 = KNeighborsClassifier(n_neighbors=3)
knn_classifier1.fit(X_train, y_train)
y_pred_class1 = knn_classifier1.predict(X_test)

knn_classifier2 = KNeighborsClassifier(n_neighbors=3)
knn_classifier2.fit(X_train, y_train)
y_pred_class2 = knn_classifier2.predict(X_test)

knn_classifier3 = KNeighborsClassifier(n_neighbors=3)
knn_classifier3.fit(X_train, y_train)
y_pred_class3 = knn_classifier3.predict(X_test)

print('KNeighborsClassifier1 : ',metrics.accuracy_score(y_test, y_pred_class1))
print('KNeighborsClassifier2 : ',metrics.accuracy_score(y_test, y_pred_class2))
print('KNeighborsClassifier3 : ',metrics.accuracy_score(y_test, y_pred_class3))
```

    KNeighborsClassifier1 :  0.9084444444444445
    KNeighborsClassifier2 :  0.9084444444444445
    KNeighborsClassifier3 :  0.9084444444444445
    

It shows that classification model has higher accuracy compare to regression model in this prediction.
I am going to use 2 for n_neighbors.
hr_predict.csv is record of current employee in the company. using trained model to predict if the employee will stay or leave.


```python
predict = pd.read_csv("hr_predict.csv")
X_predict = predict[inputs]
predictions1 = knn_regressor2.predict(X_predict)
predictions2 = knn_classifier2.predict(X_predict)
```


```python
# Storing prediction result to dataframe

percent_left= (predictions1 * 100)
predict['left_predict_percent'] = percent_left.round(2)
predict['left_predict2'] = predictions2
predict
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>names</th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_monthly_hours</th>
      <th>exp_in_company</th>
      <th>work_accident</th>
      <th>promotion_last_5years</th>
      <th>role</th>
      <th>salary</th>
      <th>salary_class</th>
      <th>role_class</th>
      <th>left_predict_percent</th>
      <th>left_predict2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2237</td>
      <td>Paul Mathey</td>
      <td>0.74</td>
      <td>0.72</td>
      <td>4</td>
      <td>176</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8127</td>
      <td>Shawn Torres</td>
      <td>0.72</td>
      <td>0.88</td>
      <td>3</td>
      <td>224</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>technical</td>
      <td>low</td>
      <td>1</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8146</td>
      <td>Emily Staples</td>
      <td>0.52</td>
      <td>0.67</td>
      <td>4</td>
      <td>216</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>support</td>
      <td>medium</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14441</td>
      <td>Jean Salazar</td>
      <td>0.42</td>
      <td>0.47</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>marketing</td>
      <td>low</td>
      <td>1</td>
      <td>6</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11909</td>
      <td>Javier Folse</td>
      <td>0.85</td>
      <td>0.58</td>
      <td>4</td>
      <td>186</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>12162</td>
      <td>Robert Davis</td>
      <td>0.45</td>
      <td>0.51</td>
      <td>2</td>
      <td>147</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
      <td>1</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>14646</td>
      <td>Kevin Reid</td>
      <td>0.41</td>
      <td>0.47</td>
      <td>2</td>
      <td>135</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>support</td>
      <td>low</td>
      <td>1</td>
      <td>3</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>10385</td>
      <td>Mary Valdez</td>
      <td>0.99</td>
      <td>0.50</td>
      <td>4</td>
      <td>173</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>management</td>
      <td>low</td>
      <td>1</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>9524</td>
      <td>William Jaeger</td>
      <td>0.74</td>
      <td>0.55</td>
      <td>5</td>
      <td>168</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>management</td>
      <td>low</td>
      <td>1</td>
      <td>11</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>13936</td>
      <td>John Booker</td>
      <td>0.16</td>
      <td>0.63</td>
      <td>6</td>
      <td>286</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>sales</td>
      <td>medium</td>
      <td>2</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 15 columns</p>
</div>



Two columns were added left_predict_percent is percentage if employee will leave company using regression model and left_predict2 is prediction if employee is going to leave using clasification model (1 = will leave, 0=will stay).


```python
sns.countplot(x="number_project", hue="left_predict2", data=predict );
```


    
![png](output_22_0.png)
    


**Final Observation and Suggestion**

The reason that I include regression model result is to show percentage that an employee is going to leave where classification model is just showing leave or stay result.

1. Based on the prediction most of people with 2 projects per year are going to leave, this may show that they may wants more challenges.
2. Three or four projects per year has the less attrition rate
3. Managers with the staff left_predict_percent = 33.33% (or greater) can have 1:1 conversation with their staff to review their workload and to understand what excites them(if they want to retain their staff)








```python
# Saving results to csv file
predict.to_csv("final_predictions.csv")
```


```python

```
