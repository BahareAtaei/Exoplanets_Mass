"""
step1:

step1: PREPROCESSING

(1) import libraries and files
(2) print all columns and shape of main datase
(3) check missing values in the first place
(4) fill semi_major_axis and orbital_period by formula then check missing values
    a lot of data(semi_major_Axis, orbital_period) will produce in this way
(5) fill columns with less missing data by median
(6) fill columns with more missing data by KNN
(7) fill eccentricity, temp_calculated by KNN too
(8) save changes to a new dataset
"""
#==========================================

# import libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
#================================================

# import file
use_cols = [
   'name',
    'star_name',
    'detection_type',
    'radius',
    'mass',
    'star_mass',
    'star_radius',
    'semi_major_axis',
    'orbital_period',
    'star_teff',
    'star_metallicity',
    'eccentricity',
    'temp_calculated'
]
print("Importing Files ...")
df1 = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A_A\exoplanet.eu_catalog_03-08-26_16_42_07.csv')
dff = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\Model_A_A\exoplanet.eu_catalog_03-08-26_16_42_07.csv', usecols=use_cols)
df = dff.copy()
print('='*50)
print('')
#===================================================================

# list of original columns
print('Columns of Original File: ')
print(df1.columns.to_list())
print('='*50)
print('')

# dimention of original file
print('Dimention of Original DataSet: ')
print(df1.shape)
print('='*50)
print('')
#==============================================

# first drop duplicated rows
print('Numbers of Duplicated Rows: ')
print(df.duplicated().sum())
print('')

print('Dimention of DataSet after Duplication: ')
df = df.drop_duplicates()
print(df.shape)
print('='*50)
print('')
#==================================================

# check missing values
print('Check Missing Values')
missing_values = pd.DataFrame(
    {
        'Missing': df.isnull().sum(),
        'Percent': (df.isnull().mean() * 100).round(2)
    }
)

missing_values = missing_values.sort_values('Percent', ascending = False)
print(missing_values)
print('='*50)
print('')
#=================================================

# fill semi_mjor_Axis and orbital_period
# print nan values of semi_major_axis and orbital_period before filling
print('Before Filling Semi_Major_Axis and Orbital_Period: ')
print(df[['semi_major_axis', 'orbital_period']].isnull().sum())
print('')

# orbital_period(days) -> orbital_period(years)
orbital_period = df['orbital_period'] / 356.25

# where "semi_major_axis" is NaN but "orbital_period" and "star_mass" aren't NaN
mask_sma = (df['semi_major_axis'].isna() & df['orbital_period'].notna() & df['star_mass'].notna())

# fill nan values in semi_major_axis by conditions of mask_a 
df.loc[mask_sma, 'semi_major_axis'] = (orbital_period[mask_sma]**2 * df.loc[mask_sma, 'star_mass'])**(1 / 3)

# where "orbital_period" is NaN but "semi_major_axis" and "star_mass" aren't NaN
mask_op = (df['orbital_period'].isna() & df['semi_major_axis'].notna() & df['star_mass'].notna())
df.loc[mask_op, 'orbital_period'] = np.sqrt(df.loc[mask_op, 'semi_major_axis']**3 / df.loc[mask_op, 'star_mass']) * 365.25

# print results
print('NaN Values of "Semi Major Axis" and "Orbital_period":  ')
print(df[['semi_major_axis', 'orbital_period']].isnull().sum())
print('='*50)
print('')
#===========================================================

# check missing values after filling semi_major_axis and orbital_period
print('Check Missing Values: After Filling Semi_Major_Axis and Orbital_Period')
missing_values = pd.DataFrame(
    {
        'Missing': df.isnull().sum(),
        'Percent': (df.isnull().mean() * 100).round(2)
    }
)
missing_values = missing_values.sort_values('Percent', ascending=False)
print(missing_values)
print('='*50)
print('')
#==================================================

# drop nan radiuses
df = df.dropna(subset=['radius'])

# check missing values after removing nan radiuses
print('Check Missing Values: After Removing NaN Radiuses')
missing_values = pd.DataFrame(
    {
        'Missing': df.isnull().sum(),
        'Percent': (df.isnull().mean() * 100).round(2)
    }
)
missing_values = missing_values.sort_values('Percent', ascending=False)
print(missing_values)
print('='*50)
print('')
#================================================


"""
FILLING METHODS ACCORDING TO UPPER MISSING VALUES OUTPUT

KNN: [
    star_metallicity  
    star_mass 
    eccentricity       
    temp_calculated 
]

Median[
    star_teff                
    star_radius             
    semi_major_axis                   
    orbital_period            
]

"""
#=======================================================

# fill columns by median
median_columns = [
    
    'star_teff',
    'star_radius',
    'semi_major_axis',
    'orbital_period'
]
for col in median_columns:
    df[col] = df[col].fillna(df[col].median())
#====================================================

# fill by KNN -> star_metallicity, star_mass
num_cols = [
    'star_mass',
    'star_metallicity',
    'eccentricity',       
    'temp_calculated'
]
standard_scaler = StandardScaler()
scaled_values = standard_scaler.fit_transform(df[num_cols])
imputer = KNNImputer(n_neighbors = 5)
imputed_scaled = imputer.fit_transform(scaled_values)
df[num_cols] = standard_scaler.inverse_transform(imputed_scaled)
#============================================================

# check the missing values for the last time
print('Check Missing Values: Last Time')
missing_values = pd.DataFrame(
    {
        'Missing': df.isnull().sum(),
        'Percent': (df.isnull().mean() * 100).round(2)
    }
)
missing_values = missing_values.sort_values('Percent', ascending=False)
print(missing_values)
print('='*50)
print('')
#===========================================================

# save changes to new dataset
df.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\FeatureImportance\preprocessed.csv', index=False)
