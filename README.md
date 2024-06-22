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
