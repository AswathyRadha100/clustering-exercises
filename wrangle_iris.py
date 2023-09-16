# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
#-------------------Imports for data manipulation and analysis --------------------- 
import pandas as pd

#-------------------Imports for numerical computations ----------------------------- 
import numpy as np

#-------------------Import custom library ------------------------------------------ 
import env

#-------------------Import SQLAlchemy-----------------------------------------------
from sqlalchemy import create_engine, text

#-------------------Import for ignore warnings--------------------------------------
import warnings
warnings.filterwarnings("ignore")

#----------------- Import the 'os' module to access operating system functionality---
import os

#-----------------Import for data visualization--------------------------------------
import matplotlib.pyplot as plt

#-----------------Import for advanced data visualization-----------------------------
import seaborn as sns

#-----------------Import for data splitting------------------------------------------
from sklearn.model_selection import train_test_split


# -

##############Function to fetch data from iris database######################
def get_iris_data():
    '''
    This function acquires iris.csv if it is available,
    otherwise, it makes the SQL connection and uses the query provided
    to read in the dataframe from SQL.
    If the CSV is not present, it will write one.
    '''
    filename = "iris.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        # Create the URL for the database connection
        url = env.get_db_url('iris_db')

        # Define the SQL query
        iris_query = '''
    SELECT *
    FROM measurements
	LEFT JOIN species USING (species_id)
    ;
    '''
        # Read the SQL query into a dataframe
        df = pd.read_sql(iris_query, url)

        # Write the dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df


def missing_value_summary(df):
    """
    Generate a summary of missing values in a DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame to analyze.

    Returns:
    DataFrame: A DataFrame with columns 'name', 'num_rows_missing', and 'pct_rows_missing'.
               'name' - Column names in the input DataFrame.
               'num_rows_missing' - Number of missing rows for each column.
               'pct_rows_missing' - Percentage of missing rows for each column.
    
    This function calculates the number and percentage of missing values for each column in the input DataFrame.
    It returns a summary DataFrame with this information.
    """
    new_columns = ['name', 'num_rows_missing', 'pct_rows_missing']
    
    new_df = pd.DataFrame(columns=new_columns)
    
    for col in list(df.columns):
        num_missing = df[col].isna().sum()
        pct_missing = num_missing / df.shape[0]
        
        add_df = pd.DataFrame([{'name': col, 'num_rows_missing': num_missing,
                               'pct_rows_missing': pct_missing}])
        
        new_df = pd.concat([new_df, add_df], axis=0)
        
    new_df.set_index('name', inplace=True)
    return new_df



# +
def summarize(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    
    info
    shape
    outliers
    description
    missing data stats
    
    return: None (prints to console)
    '''
    # print info on the df
    print('Shape of Data: ')
    print(df.shape)
    print('======================\n======================')
    print('Info: ')
    print(df.info())
    print('======================\n======================')
    print('Descriptions:')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('======================\n======================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('O').describe().T.to_markdown())
    print('======================\n======================')
    print('missing values:')
    print('by column:')
    print(missing_by_col(df).to_markdown())
    print('by row: ')
    print(missing_by_row(df).to_markdown())
    print('======================\n======================')
    print('Outliers: ')
    print(report_outliers(df, k=k))
    print('======================\n======================')

def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    return df.isnull().sum(axis=0)

def missing_by_row(df):
    '''
    prints out a report of how many rows have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)
    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be synonymous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
        'num_cols_missing': count_missing,
        'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')

def get_fences(df, col, k=1.5) -> tuple:
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    
    return: lower_bound and upper_bound, two floats
    '''
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound

    
