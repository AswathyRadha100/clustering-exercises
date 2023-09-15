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
#------------------- Tabular data imports--------------------- 
import pandas as pd

import numpy as np

#-------------------Import custom library --------------------- 
import env

#-------------------Import SQLAlchemy--------------------------
from sqlalchemy import create_engine, text

#-------------------ignore warnings----------------------------
import warnings
warnings.filterwarnings("ignore")

#----------------- Import the 'os' module to access operating system functionality---------
import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split


# +
# ##############Function to fetch data from zillow database######################
# def get_zillow_data():
#     '''
#     This function acquires zillow.csv if it is available,
#     otherwise, it makes the SQL connection and uses the query provided
#     to read in the dataframe from SQL.
#     If the CSV is not present, it will write one.
#     '''
#     filename = "zillow.csv"

#     if os.path.isfile(filename):
#         return pd.read_csv(filename, index_col=0)
#     else:
#         # Create the URL for the database connection
#         url = env.get_db_url('zillow')

#         # Define the SQL query
#         zillow_query = """
#     SELECT prop.*, 
#        pred.logerror, 
#        pred.transactiondate, 
#        air.airconditioningdesc, 
#        arch.architecturalstyledesc, 
#        build.buildingclassdesc, 
#        heat.heatingorsystemdesc, 
#        landuse.propertylandusedesc, 
#        story.storydesc, 
#        construct.typeconstructiondesc 
#     FROM   properties_2017 prop  
#        INNER JOIN (SELECT parcelid,
#        					  logerror,
#                           Max(transactiondate) transactiondate 
#                    FROM   predictions_2017 
#                    GROUP  BY parcelid, logerror) pred
#                USING (parcelid) 
#        LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
#        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
#        LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
#        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
#        LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
#        LEFT JOIN storytype story USING (storytypeid) 
#        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
#     WHERE  prop.latitude IS NOT NULL 
#        AND prop.longitude IS NOT NULL
#     """

#         # Read the SQL query into a dataframe
#         df = pd.read_sql(zillow_query, url)

#         # Write the dataframe to disk for later. Called "caching" the data for later.
#         df.to_csv(filename)

#         # Return the dataframe to the calling code
#         return df

# -

##############Function to fetch data from zillow database######################
def get_zillow_data():
    '''
    This function acquires zillow.csv if it is available,
    otherwise, it makes the SQL connection and uses the query provided
    to read in the dataframe from SQL.
    If the CSV is not present, it will write one.
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        # Create the URL for the database connection
        url = env.get_db_url('zillow')

        # Define the SQL query
        zillow_query = """
    SELECT prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        land.propertylandusedesc,
        story.storydesc,
        type.typeconstructiondesc
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
    """

        # Read the SQL query into a dataframe
        df = pd.read_sql(zillow_query, url)

        # Write the dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

# # Alternative method with temp table

# +
# Import the necessary libraries
# import sqlalchemy
# import pandas as pd

# Define your database connection details
# Use your database name, username, password, and host
#database = '**your database name**'
#user = 'your_username'
#password = 'your_password'
#host = 'your_host'

# Create a database engine for connecting to the MySQL database
# engine = sqlalchemy.create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

# Establish a database connection
# connection = engine.connect()

# Create a temporary table in the database
# temp_table = '''
#     create temporary table pred_agg
#     SELECT parcelid, MAX(transactiondate) AS max_transactiondate, logerror
#     FROM zillow.predictions_2017
#     GROUP BY parcelid, logerror
# '''

# Execute the SQL statement to create the temporary table
# connection.execute(sqlalchemy.text(temp_table))

# Define a query to retrieve data from multiple tables
# query = '''
#     select *
#     from pred_agg
#     left join zillow.properties_2017 using(parcelid)
#     left join zillow.airconditioningtype using(airconditioningtypeid)
#     left join zillow.architecturalstyletype using(architecturalstyletypeid)
#     left join zillow.buildingclasstype using(buildingclasstypeid)
#     left join zillow.heatingorsystemtype using(heatingorsystemtypeid)
#     left join zillow.propertylandusetype using(propertylandusetypeid)
#     left join zillow.storytype using(storytypeid)
#     left join zillow.typeconstructiontype using(typeconstructiontypeid)
#     WHERE propertylandusedesc = "Single Family Residential"
#     	and longitude is not null
#         and latitude is not null;
# '''

# Execute the query and load the results into a pandas DataFrame
# df = pd.read_sql(query, connection)

# Display the first few rows of the DataFrame
# df.head()


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

    


# +

def missing_value_summary(df):
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
def display_numeric_column_histograms(data_frame):
    """
    Display histograms for numeric columns in a DataFrame with three colors.

    Args:
    - data_frame: pandas DataFrame

    Returns:
    None
    """
    numeric_columns = data_frame.select_dtypes(exclude=["object", "category"]).columns.to_list()
    
    # Define two colors for the histogram bars
    colors = ["m", "orange","olive"]
    
    for i, column in enumerate(numeric_columns):
        # Create a histogram for each numeric column with two colors
        figure, axis = plt.subplots(figsize=(10, 3))
        sns.histplot(data_frame, x=column, ax=axis, color=colors[i % len(colors)])
        axis.set_title(f"Histogram of {column}")
        plt.show()



# +

def handle_missing_values(df, prop_required_column, prop_required_row):
    """
    Remove columns and rows based on the proportion of missing values.

    Parameters:
    - df: DataFrame
    - prop_required_column: float (default=0.6)
      Proportion of non-missing values required for columns.
    - prop_required_row: float (default=0.75)
      Proportion of non-missing values required for rows.

    Returns:
    - DataFrame with columns and rows dropped as indicated.
    """
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    row_threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    
    return df



# -

def filter_single_unit_properties(df):
    """
    Filter out properties that are likely not single-unit properties based on property types.

    Parameters:
    - df: DataFrame

    Returns:
    - DataFrame containing only single-unit properties.
    """
    # List of property types considered as single-unit properties
    single_unit_types = ['Single Family', 'Condo', 'Townhouse']
    
    # Filter the DataFrame to include only rows with property types in the list
    filtered_df = df[df['property_type'].isin(single_unit_types)]
    
    return filtered_df


# +
#def split_data(df, target=None) -> tuple:
#    '''
#    split_data will split data into train, validate, and test sets
    
#    if a discrete target is in the data set, it may be specified
#    with the target kwarg (Default None)
    
#    return: three pandas DataFrames
#    '''
#    train_val, test = train_test_split(
#        df, 
#        train_size=0.8, 
#       stratify=target)
#    train, validate = train_test_split(
#        train_val,
#        train_size=0.7,
#        random_state=1349,
#        stratify=target)
#    return train, validate, test

# +

def split_data(df):
    '''
    split_data will split data into train, validate, and test sets
    
    return: three pandas DataFrames
    '''
    train_val, test = train_test_split(
        df, 
        train_size=0.8, 
        random_state=1349)
    train, validate = train_test_split(
        train_val,
        train_size=0.7,
        random_state=1349)
    return train, validate, test



# -

def wrangle_zillow(summarization=True, k=1.5) -> tuple:
    '''
    wrangle_mall will acquire and prepare mall customer data
    
    if summarization is set to True, a console report 
    of data summary will be output to the console.
    
    return: train, validate, and test data sets with scaled numeric information
    '''
    if summarization:
        summarize(acquire_mall(), k=k)
    train, validate, test = prep_mall(acquire_mall())
    return train, validate, test


# +
############################## PREPARE ZILLOW FUNCTION ##############################

def prep_zillow(df):
    '''
    This function takes in a dataframe
    renames the columns, drops nulls values in specific columns,
    changes datatypes for appropriate columns, and renames fips to actual county names.
    Then returns a cleaned dataframe
    '''
    # Rename columns
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'area',
        'taxvaluedollarcnt': 'taxvalue',
        'fips': 'county',
        'lotsizesquarefeet': 'lotsqft'
    })

    # Drop specific columns
    df = df.drop(['taxamount'], axis=1)

    # Drop rows with null values in specific columns
    columns_with_nulls = ['bedrooms', 'bathrooms', 'area', 'taxvalue', 'yearbuilt', 'lotsqft']
    df = df.dropna(subset=columns_with_nulls)

    # Change data types to integers for appropriate columns
    make_ints = ['bedrooms', 'area', 'taxvalue', 'yearbuilt', 'lotsqft']

    for col in make_ints:
        df[col] = df[col].astype(int)

    # Map county codes to county names 
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    # Create dummy variables for the 'county' column with integer data type
    dummies = pd.get_dummies(df['county'],dtype=int)
    # Concatenate the dummy variables with the original dataframe
    df = pd.concat([df, dummies], axis=1)
     
        
    # Convert Column Names to Lowercase and Replace Spaces with Underscores
    df.columns = map(str.lower,df.columns)
    df.columns = df.columns.str.replace(' ','_')    
    
    return df
# -


