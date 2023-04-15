import json
import os

import plotly
import plotly.graph_objs as go
from plotly.graph_objs import layout
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, send_from_directory
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from main import df_avg, scaler, X

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               '/static/icon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    # Get the unique states and crops from the dataset
    df = pd.read_csv("cleaned_crop_production.csv")
    states = sorted(df['State'].unique())
    crops = sorted(df['Crop'].unique())
    seasons = sorted(df['Season'].unique())
    return render_template('home.html', states=states, crops=crops, seasons=seasons)

@app.route('/predict')
def predict():
    df = pd.read_csv("cleaned_crop_production.csv")
    states = sorted(df['State'].unique())
    crops = sorted(df['Crop'].unique())
    seasons = sorted(df['Season'].unique())
    return render_template('predict.html',states=states, crops=crops, seasons=seasons)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

@app.route('/result', methods=['POST'])
def result():
    try:
        state = request.form['state']
        crop = request.form['crop']
        season = request.form['season']

        # Load the models
        production_model = joblib.load('production_model.joblib')
        yield_model = joblib.load('yield_model.joblib')

        # Load the data
        df = pd.read_csv("cleaned_crop_production.csv")


        # Create dummy variables for user input and fill missing columns with zeros
        X_user = pd.DataFrame({'State_' + state: [1], 'Crop_' + crop: [1], 'Season_' + season: [1]})
        X_user = X_user.reindex(columns=X.columns, fill_value=0)

        # Standardize the user input data

        X_user = scaler.transform(X_user)

        rel_production = production_model.predict(X_user)
        rel_yield = yield_model.predict(X_user)

        avg_production = df_avg.loc[
            (df_avg['State'] == state) & (df_avg['Crop'] == crop) & (df_avg['Season'] == season), 'AvgProduction'].values[0]
        avg_yield = \
        df_avg.loc[(df_avg['State'] == state) & (df_avg['Crop'] == crop) & (df_avg['Season'] == season), 'AvgYield'].values[
            0]
        predicted_production = round(avg_production / 11)
        predicted_yield = avg_yield * rel_yield

        print(predicted_production)
        print(round(i,2) for i in predicted_yield)
        results = {}
        results['crop'] = crop
        results['yield_prediction'] = predicted_yield
        results['production_prediction'] = predicted_production
        results['production_unit'] = 'kg'

        filtered_df = df[(df['State'] == state) & (df['Crop'] == crop) & (df['Season'] == season)]
        filtered_df['Year'] = filtered_df['Year'].apply(lambda x: x.split('-')[0])
        max_year = filtered_df['Year'].max()
        min_year = str(int(max_year) - 10)
        recent_df = filtered_df[(filtered_df['Year'] >= min_year) & (filtered_df['Year'] <= max_year)]

        # Create a Plotly figure object
        fig = make_subplots()
        fig1 = make_subplots()
        # Define the data for the line graph
        trace1 = go.Scatter(x=recent_df['Year'], y=recent_df['Yield'], mode='lines', name='Yield')
        trace2 = go.Scatter(x=recent_df['Year'], y=recent_df['Production'], mode='lines', name='Production')

        # Add the trace to the figure object
        fig.add_trace(trace1)
        fig1.add_trace(trace2)

        # Define the layout of the line graph
        fig.update_layout(
            title='Yield of {} in {} for {} season'.format(crop, state, season),
            xaxis_title='Year',
            yaxis_title='Yield (tonnes/hectare)',
        )
        fig1.update_layout(
            title='Production of {} in {} for {} season'.format(crop, state, season),
            xaxis_title='Year',
            yaxis_title='Production (tonnes)',
        )
        graph = go.Figure(data=fig, layout=layout)
        graph1 = go.Figure(data=fig1,layout=layout)
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON1 = json.dumps(graph1, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('result.html', crop = crop, yield_prediction = round(predicted_yield[0],2),production_prediction = predicted_production,graphJSON=graphJSON,graphJSON1=graphJSON1)
    except:
        state = request.form['state']
        crop = request.form['crop']
        season = request.form['season']
        return render_template('result1.html', crop = crop.upper(), state=state.upper(),season=season.upper())


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
