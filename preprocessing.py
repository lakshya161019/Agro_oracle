import pandas as pd
# load the data
df = pd.read_csv('CROP MAJOR.csv')

# drop unnecessary columns
df.drop(['District'], axis=1, inplace=True)

# replace 'NA' values with NaN
df.replace('NA', pd.NA, inplace=True)

# convert Area to numeric and replace negative values with NaN
df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
df.loc[df['Area'] < 0, 'Area'] = pd.NA

# convert Production to numeric and replace negative values with NaN
df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
df.loc[df['Production'] < 0, 'Production'] = pd.NA

# group by State_Name, Crop, Season and Year
grouped = df.groupby(['State', 'Crop', 'Season', 'Year'])

# sum the Area and Production for each group
grouped_sum = grouped.sum().reset_index()

# calculate the Yield and add it as a new column
grouped_sum['Yield'] = grouped_sum['Production'] / grouped_sum['Area']

# drop the original columns and keep only the cleaned data
cleaned_data = grouped_sum[['State', 'Crop', 'Season', 'Year', 'Area', 'Production', 'Yield']]

# save the cleaned data to a new csv file
cleaned_data.to_csv('cleaned_crop_production.csv', index=False)
