"""
step2: 
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance

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
# import file
dff = pd.read_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\FeatureImportance\preprocessed.csv', usecols=use_cols)
df = dff.copy()
#===================================================

# turn radius and mass scale to log10
df['log_mass'] = np.log10(df['mass'])
df['log_radius'] = np.log10(df['radius'])
#==============================================

# feature engineering
# orbital features
df['orbital_velocity'] = 2 * np.pi * df['semi_major_axis'] / df['orbital_period']
df['orbital_frequency'] = 1 / df['orbital_period'] 
df['kepler_ratio'] = (df['orbital_period'] ** 2) / (df['semi_major_axis'] ** 3)
df['angular_momentum'] = df['semi_major_axis'] * df['orbital_velocity']  

# star features
df['star_density'] = df['star_mass'] / (df['star_radius'] ** 3)
df['star_compactness'] = df['star_mass'] / df['star_radius']

# ratios of star/orbit
df['star_mass_orbit_ratio'] = df['star_mass'] / df['semi_major_axis']
df['star_radius_orbit_ratio'] = df['star_radius'] / df['semi_major_axis']

# temperature features
df['temp_ratio'] = df['temp_calculated'] / df['star_teff']
df['thermal_equilibrium'] = (df['star_teff'] ** 4) / (df['semi_major_axis'] ** 2)

# log features
df['log_period'] = np.log10(df['orbital_period'])
df['log_semi_major'] = np.log10(df['semi_major_axis'])
df['log_star_mass'] = np.log10(df['star_mass'])
df['log_star_teff'] = np.log10(df['star_teff'])

# radius features
df['radius_normalized'] = df['radius'] / df['star_radius']  
df['radius_period_ratio'] = df['radius'] / df['orbital_period']
#=================================================

features = [

    # main features
    'log_radius',               
    'star_mass',
    'star_radius',
    'semi_major_axis',
    'orbital_period',
    'star_teff',
    'star_metallicity',
    'eccentricity',             
    'temp_calculated',          
    
    # orbital features
    'orbital_velocity',#
    'kepler_ratio',#
    'orbital_frequency',#
    
    # star features
    'star_density', #
    'star_compactness', #
    
    # ratios of star/orbit
    'star_mass_orbit_ratio',#
    'star_radius_orbit_ratio',#
    
    # temperature features
    'temp_ratio',
    'thermal_equilibrium',#
    
    # log features
    'log_period',
    'log_semi_major',
    'log_star_mass',
    'log_star_teff',
    
    # radius features
    'radius_normalized',
    'radius_period_ratio'
]
#===================================================

# split nan mass and not nan mass values to two dataset
known = df[df['mass'].notna()].copy()
unknown = df[df['mass'].isnull()].copy()
#===========================================

#split values
X = known[features]
y = known['log_mass']
#===========================================



# train and test variables
RANDOM_STATE = 45
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
#==============================================

# making random forest model
model = RandomForestRegressor(
    n_estimators = 400,
    max_depth = 15,
    min_samples_leaf = 2,
    random_state =RANDOM_STATE,
    n_jobs = -1,
    min_samples_split=2,
)
#===============================================

model.fit(X_train, y_train)
#===============================================

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=45
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": result.importances_mean
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("Important Features")
print(importance)
print('-'*50)
print('')
#==================================================

# save importances features to a csv file
importance.to_csv(r'C:\Users\Sisto\Documents\Programming\Python\Projects\Exoplanets\RandomForestRegressor\FeatureImportance\importance.csv', index=False)
#===================================================

# report results to user
string = """
ACCORDING TO THE RESULT OF PERMUTATION_IMPORTANCE
I SPLIT DATA FOR RANDOMFOREST MODEL TO 3 MODEL
----------------------------------------------
MODEL 1: ALL FEATUES

MODEL_A:
FEATURES [
    'log_radius',
    'kepler_ratio',
    'star_teff',
    'orbital_velocity',
    'thermal_equilibrium',
    'star_metallicity',
    'star_compactness',
    'radius_period_ratio',
    'log_semi_major',
    'star_radius',
    'star_mass_orbit_ratio',
    'orbital_frequency',
    'star_density',
    'radius_normalized'      
]

MODEL_A_A
FEATURES [
        'log_radius',
    'kepler_ratio',
    'star_teff',
    'orbital_velocity',
    'thermal_equilibrium',
    'star_metallicity',
    'star_compactness',
    'radius_period_ratio',
    'log_semi_major',
    'star_radius',
    'star_mass_orbit_ratio',
    'orbital_frequency',
    'star_density',
    'radius_normalized',
    'eccentricity',
    'temp_calculated'
]
---------------------------------------------------------
MODEL 2: MORE IMPORTANT FEATURES

MODEL_B:
FEATURES [
    'log_radius',
    'kepler_ratio',
    'star_teff',
    'orbital_velocity',
    'thermal_equilibrium',
    'star_metallicity',
    'star_compactness',
    'radius_period_ratio',
    'log_semi_major'
]

MODEL_B_B:
FEATURES [
    'log_radius',
    'kepler_ratio',
    'star_teff',
    'orbital_velocity',
    'thermal_equilibrium',
    'star_metallicity',
    'star_compactness',
    'radius_period_ratio',
    'log_semi_major',
    'eccentricity',
    'temp_calculated'
]
-------------------------------------
MODEL 3: MOST IMPORTANT FEATURES

MODEL_C
FEATURES [
    "log_radius",
    "kepler_ratio",
    "star_teff",
    "orbital_velocity",
    "thermal_equilibrium"
]


MODEL_C_C
FEATURES [
    'log_radius',
    'kepler_ratio',
    'star_teff',
    'orbital_velocity',
    'thermal_equilibrium',
    'eccentricity',
    'temp_calculated'
]
"""
print(string)
