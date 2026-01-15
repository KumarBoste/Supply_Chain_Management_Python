# DAY - 1 
# Import Data Manipulation Libraries
import numpy as np
import pandas as pd

# Import Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import Filter Warning Libraries
import warnings
warnings.filterwarnings('ignore')


# Import Logging
import logging
logging.basicConfig(level = logging.INFO,
                    filename = 'supplychain.log',
                    filemode = 'w',
                    format = '%(asctime)s - %(message)s - %(levelname)s',
                    force = True)


# Import OrderDict Function
from collections import OrderedDict


# STEP - 1 :  Data Ingestion = Load

def data_ingesion():

  try:
    df = pd.read_csv(r'/content/SupplyChain_Dataset.csv')
    logging.info("Dataset Sucessfully Uploaded")


  except :
    logging.info("Check the location of file")

  return df



# STEP - 2 Data Exploration
def data_exploration(df):

  # Segregate Numerical and Categorical Columns
  numerical_col = df.select_dtypes(exclude = 'object').columns
  categorcal_col = df.select_dtypes(include = 'object').columns

  # Numerical Descriptive Stats
  numerical_stats = []

  Q1 = df[numerical_col].quantile(0.25)
  Q3 = df[numerical_col].quantile(0.75)
  IQR = Q3 - Q1
  LW = Q1 - 1.5*IQR
  UW = Q3 + 1.5*IQR
  Outlier_Count = (df[numerical_col] < LW)  | (df[numerical_col] > UW)
  Outlier_Percentage = (Outlier_Count.sum()/len(df)*100)

  for i in numerical_col:
    num_stats = OrderedDict({
        "Feature" :i,
        "Count": df[i].count(),
        "Maximum": df[i].max(),
        "minimum":df[i].min(),
        "Mean":df[i].mean(),
        "Median": df[i].median(),
        "Q1":Q1,
        "Q3":Q3,
        "IQR":IQR,
        "Lower Whisker":LW,
        "Upper Whisker": UW,
        "Outlier_Count":Outlier_Count.sum(),
        "Outlier_Percentage":Outlier_Percentage,
        "Skewness":df[i].skew(),
        "Kurtosis":df[i].kurtosis(),
        "Standard Deviation":df[i].std()

    })

    numerical_stats.append(num_stats)
    numerical_stats_report = pd.DataFrame(numerical_stats)

 # Categorical Descriptive Stats

  categorical_stats = []

  for i in  categorcal_col:
    cat_stats = OrderedDict({
        "Feature":i,
        "Count":df[i].count(),
        "Unique_Count":df[i].nunique(),
        "Mode":df[i].mode(),
        "Value_Counts":df[i].value_counts()
    })
    categorical_stats.append(cat_stats)
    categorical_stats_report = pd.DataFrame(categorical_stats)

  return numerical_stats_report,categorical_stats_report


# STEP - 3 : Data Information
def dataset_info(df):
  print(df.info())
# ----------------------------------------------------------------------------------------
'''
def main() :
  df = data_ingesion()
  numerical_stats_report, categorical_stats_report = data_exploration(df)

  return numerical_stats_report,categorical_stats_report


main()
'''

df = data_ingesion()
numerical_stats_report,categorical_stats_report = data_exploration(df)
info = dataset_info(df)


# Data Cleaning

# Checking Duplicate Entries
df.duplicated().sum()

# This will help us to drop the duplicated entries from the data
df.drop_duplicates(inplace = True)

# Check the missing values in the dataset
plt.figure(figsize=(10,15))
df.isna().sum().plot(kind= 'barh')
plt.title("Missing Values in SupplyChain Dataset")
plt.show()

# Dropping Product_Description and Order_Zipcode columns as these columns how more number of missing
df.drop(columns=['Product_Description','Order_Zipcode'], axis = 1, inplace= True)

## It giving name of Columns
df.columns

# Checking Null values in the columns 
df.isna().sum()


# Use for Display all the Columns
pd.set_option("display.max_columns",None)
df


# Data Preprocessing
# Change DataType of Two Columns
df["order_date_(DateOrders)"] = pd.to_datetime(df["order_date_(DateOrders)"])
df["shipping_date_(DateOrders)"] = pd.to_datetime(df["shipping_date_(DateOrders)"])

df["order_month"] = df["order_date_(DateOrders)"].dt.month
df["order_year"] = df["order_date_(DateOrders)"].dt.year
df["order_day"] = df["order_date_(DateOrders)"].dt.day

df["shipping_month"] = df["shipping_date_(DateOrders)"].dt.month
df["shipping_year"] = df["shipping_date_(DateOrders)"].dt.year
df["shipping_day"] = df["shipping_date_(DateOrders)"].dt.day



plt.figure(figsize=(12,10))

plt.subplot(1,2,1)
df["order_year"].value_counts().plot(kind='pie',
                                     autopct = '%0.2f%%',
                                     explode = [0.04,0.04,0.04,0.04],
                                     legend =['2015','2016','2017','2018']
                                     )

plt.title("Year Wise Percentage of Order Placed")


plt.subplot(1,2,2)
df["shipping_year"].value_counts().plot(kind='pie',
                                     autopct = '%0.2f%%',
                                     explode = [0.04,0.04,0.04,0.04],
                                     legend =['2015','2016','2017','2018']
                                     )

plt.title("Year Wise Percentage of Shipping Order")
plt.legend()
plt.show()


# Which Country Showing hightest number of Sales

ct1 = pd.crosstab(index = [df["Order_Country"], df["Order_City"]],
                           columns = [df["Product_Name"],df["Order_Profit_Per_Order"]],
                  margins = True)
ct1
