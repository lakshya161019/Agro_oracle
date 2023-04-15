import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the cleaned CSV file
df = pd.read_csv("cleaned_crop_production.csv")

# Group districts by state, crop and season
df_grouped = df.groupby(['State', 'Crop', 'Season']).agg({'Area': 'sum', 'Production': 'sum', 'Yield': 'mean'}).reset_index()

# Compute the historical average production, yield, and area for each state, crop, and season
df_avg = df_grouped.groupby(['State', 'Crop', 'Season']).agg({'Production': 'mean', 'Yield': 'mean', 'Area': 'mean'}).reset_index()
df_avg.columns = ['State', 'Crop', 'Season', 'AvgProduction', 'AvgYield', 'AvgArea']

# Merge the historical average with the grouped data
df_grouped = pd.merge(df_grouped, df_avg, on=['State', 'Crop', 'Season'], how='left')

# Compute the relative production, yield, and area for each district based on the historical average
df_grouped['RelProduction'] = df_grouped['Production'] / df_grouped['AvgProduction']
df_grouped['RelYield'] = df_grouped['Yield'] / df_grouped['AvgYield']
df_grouped['RelArea'] = df_grouped['Area'] / df_grouped['AvgArea']

# Convert State, Crop, and Season to categorical variables
df_grouped['State'] = pd.Categorical(df_grouped['State'])
df_grouped['Crop'] = pd.Categorical(df_grouped['Crop'])
df_grouped['Season'] = pd.Categorical(df_grouped['Season'])

# Convert categorical variables to dummy variables
df_grouped = pd.get_dummies(df_grouped, columns=['State', 'Crop', 'Season'])

# Split the data into training and testing sets
X = df_grouped.drop(['Production', 'Yield', 'AvgProduction', 'AvgYield', 'AvgArea', 'RelProduction', 'RelYield', 'RelArea'], axis=1)
y1 = df_grouped['RelProduction']
y2 = df_grouped['RelYield']
y3 = df_grouped['RelArea']
y1.fillna(y1.mean(), inplace=True)
y2.fillna(y2.mean(), inplace=True)
y3.fillna(y3.mean(), inplace=True)

X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X, y1, y2, y3, test_size=0.2, random_state=42)

# Standardize the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the random forest regression model to predict relative production
production_model = RandomForestRegressor(n_estimators=100, random_state=42)
production_model.fit(X_train, y1_train)

# Train the random forest regression model to predict relative yield
yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model.fit(X_train, y2_train)

# Train the random forest regression model to predict relative area
area_model = RandomForestRegressor(n_estimators=100, random_state=42)
area_model.fit(X_train, y3_train)

# Save the trained models
joblib.dump(production_model, 'production_model.joblib')
joblib.dump(yield_model,'yield_model.joblib')
joblib.dump(area_model,'area_model.joblib')