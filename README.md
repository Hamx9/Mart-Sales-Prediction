# Big Mart Sales Prediction

## Project Scope

Having a well-defined structure before performing a task helps in efficient execution of the task. This is true even in cases of building a machine learning model. Once you have built a model on a dataset, you can easily break down the steps and define a structured Machine learning pipeline.

This notebook covers the process of building an end-to-end Machine Learning pipeline and implementing it on the BigMart sales prediction dataset.

The dataset contains information about the stores, products and historical sales. We will predict the sales of the products in the stores.

We will start by building a prototype machine learning pipeline that will help us define the actual machine learning pipeline.

## Importing Libraries

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Data Exploration and Preprocessing

### Loading the Train Data

```python
train = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
```

### Checking for Missing Values

```python
train.isna().sum()
```

```
Item_Identifier                 0
Item_Weight                  1463
Item_Fat_Content                0
Item_Visibility                 0
Item_Type                       0
Item_MRP                        0
Outlet_Identifier               0
Outlet_Establishment_Year       0
Outlet_Size                  2410
Outlet_Location_Type            0
Outlet_Type                     0
Item_Outlet_Sales               0
dtype: int64
```

Only `Item_Weight` and `Outlet_Size` have missing values.

`Item_Weight` is a continuous variable, so we can use the mean to impute the missing values.

`Outlet_Size` is a categorical variable, so we will use the mode to impute the missing values.

### Imputing Missing Values

```python
# Impute missing values in Item_Weight using mean
train.Item_Weight.fillna(train.Item_Weight.mean(), inplace=True)
train.Item_Weight.isna().sum()
# 0

# Impute missing values in Outlet_Size using mode
train.Outlet_Size.fillna(train.Outlet_Size.mode()[0], inplace=True)
train.Outlet_Size.isna().sum()
# 0
```

### Converting Categorical Variables to Numeric

Machine learning models cannot work with categorical (string) data. We will convert the categorical variables into numeric types.

```python
# Checking categorical variables in the data
train.dtypes
```

```
Item_Identifier               object
Item_Weight                  float64
Item_Fat_Content              object
Item_Visibility              float64
Item_Type                     object
Item_MRP                     float64
Outlet_Identifier             object
Outlet_Establishment_Year      int64
Outlet_Size                   object
Outlet_Location_Type          object
Outlet_Type                   object
...
```



Our data has the following categorical variables:

- Item_Identifier
- Item_Fat_Content
- Item_Type
- Outlet_Identifier
- Outlet_Size
- Outlet_Type
- Outlet_Location_Type

We will use the `categorical_encoders` library to convert these variables into binary variables. We will not convert `Item_Identifier`.


## One-Hot Encoding Categorical Variables

```python
import category_encoders as ce

# Create an object of OneHotEncoder
OHE = ce.OneHotEncoder(cols=['Item_Fat_Content',
                            'Item_Type',
                            'Outlet_Identifier',
                            'Outlet_Size',
                            'Outlet_Location_Type',
                            'Outlet_Type'],use_cat_names=True)

# Encode the variables
train = OHE.fit_transform(train)
```
/opt/conda/lib/python3.7/site-packages/category_encoders/utils.py:21: FutureWarning: is_categorical is deprecated and will be removed in a future version.  Use is_categorical_dtype instead
  elif pd.api.types.is_categorical(cols):
```python
train.head()
```

| Item_Identifier | Item_Weight | Item_Fat_Content_Low Fat | Item_Fat_Content_Regular | Item_Fat_Content_low fat | Item_Fat_Content_LF | Item_Fat_Content_reg | Item_Visibility | Item_Type_Dairy | Item_Type_Soft Drinks | ... | Outlet_Size_High | Outlet_Size_Small | Outlet_Location_Type_Tier 1 | Outlet_Location_Type_Tier 3 | Outlet_Location_Type_Tier 2 | Outlet_Type_Supermarket Type1 | Outlet_Type_Supermarket Type2 | Outlet_Type_Grocery Store | Outlet_Type_Supermarket Type3 | Item_Outlet_Sales |
|-----------------|-------------|--------------------------|-------------------------|-------------------------|-------------------|---------------------|-----------------|-----------------|----------------------|-----|-----------------|-------------------|----------------------------|----------------------------|----------------------------|------------------------------|------------------------------|-------------------------|-----------------------------|--------------------|
| FDA15           | 9.30        | 1                        | 0                       | 0                       | 0                 | 0                   | 0.016047        | 1               | 0                     | ... | 0                | 0                 | 1                          | 0                          | 0                          | 1                           | 0                           | 0                        | 0                           | 3735.1380         |
| DRC01           | 5.92        | 0                        | 1                       | 0                       | 0                 | 0                   | 0.019278        | 0               | 1                     | ... | 0                | 0                 | 0                          | 1                          | 0                          | 0                           | 1                           | 0                        | 0                           | 443.4228          |
| FDN15           | 17.50       | 1                        | 0                       | 0                       | 0                 | 0                   | 0.016760        | 0               | 0                     | ... | 0                | 0                 | 1                          | 0                          | 0                          | 1                           | 0                           | 0                        | 0                           | 2097.2700         |
| FDX07           | 19.20       | 0                        | 1                       | 0                       | 0                 | 0                   | 0.000000        | 0               | 0                     | ... | 0                | 0                 | 0                          | 1                          | 0                          | 0                           | 0                           | 1                        | 0                           | 732.3800          |
| NCD19           | 8.93        | 1                        | 0                       | 0                       | 0                 | 0                   | 0.000000        | 0               | 0                     | ... | 1                | 0                 | 0                          | 1                          | 0                          | 1                           | 0                           | 0                        | 0                           | 994.7052          |

5 rows × 47 columns


Now that we have taken care of our categorical variables, we move on to the continuous variables. We will normalize the data in such a way that the range of all variables is almost similar. We will use the `StandardScaler` function to do this.

```python
from sklearn.preprocessing import StandardScaler
# create an object of the StandardScaler
scaler = StandardScaler()

# fit with the Item_MRP
scaler.fit(np.array(train.Item_MRP).reshape(-1,1))

# transform the data
train.Item_MRP = scaler.transform(np.array(train.Item_MRP).reshape(-1,1))
```

# Building the Model

We will use the Linear Regression and the Random Forest Regressor to predict the sales. We will create a validation set using the `train_test_split()` function.

`test_size = 0.25` such that the validation set holds 25% of the data points while the train set has 75%.

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# separate the independent and target variable
train_X = train.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'])
train_Y = train['Item_Outlet_Sales']

# split the data
train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y, test_size=0.25)

# shape of train test splits
train_x.shape, valid_x.shape, train_y.shape, valid_y.shape
# ((6392, 45), (2131, 45), (6392,), (2131,))
```

Now that we have split our data, we will train a linear regression model on this data and check its performance on the validation set. We will use RMSE as an evaluation metric.

```python
# LinearRegression
LR = LinearRegression()

# fit the model
LR.fit(train_x, train_y)

# predict the target on train and validation data
train_pred = LR.predict(train_x)
valid_pred = LR.predict(valid_x)

# RMSE on train and validation data
print('RMSE on train data: ', mean_squared_error(train_y, train_pred)**(0.5))
print('RMSe on validation data: ', mean_squared_error(valid_y, valid_pred)**(0.5))
# RMSE on train data:  1120.746601859512
# RMSe on validation data:  1147.9427065958134
```

We will train a random forest regressor and see if we can get an improvement on the train and validation errors.


# Random Forest Regressor

```python
#RandomForestRegressor
RFR = RandomForestRegressor(max_depth=10)

#fitting the model
RFR.fit(train_x, train_y)

#predict the target on train and validation data
train_pred = RFR.predict(train_x)
valid_pred = RFR.predict(valid_x)

#RMSE on train and test data
print('RMSE on train data :', mean_squared_error(train_y, train_pred)**(0.5))
print('RMSE on validation data :', mean_squared_error(valid_y, valid_pred)**(0.5))
```

Output:
```
RMSE on train data : 894.6959458326626
RMSE on validation data : 1107.415312588672
```

We can see a significant improvement on the RMSE values. The random forest algorithm gives us 'feature importance' for all the variables in the data.

We have 45 features and not all of these features may be useful in forecasting. We will select the top 7 features which had a major contribution in forecasting sales values.

If the model performance is similar in both cases (by using 45 features and by using 7 features), then we should only use the top 7 features, in order to keep the model simple and efficient.

The goal is to have a less complex model without compromising on the overall model performance.

```python
#plot the 7 most important features
plt.figure(figsize=(10,8))
feat_importances = pd.Series(RFR.feature_importances_, index = train_x.columns)
feat_importances.nlargest(7).plot(kind='barh');
```
