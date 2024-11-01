from flask import Flask, render_template, request, redirect, url_for,flash
from disease_scraper import scrape_disease_info
from custom_model_manager import *
from hospital_loader import *
import os
import pandas as pd 
import numpy as np 
import xgboost as xgb
import pickle
import json
import matplotlib
import os
import requests
app = Flask(__name__)


# Configurations
CONFIG_FILE_PATH = 'front end/config/config.json'
MODELS_DIR = 'Back end/chk_pts/'
SHAP_VALUES_DIR = 'Back end/shap_values/'
SHAP_PLOT_DIR = 'front end/static/shap_plots/'
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'Back end\ml_data'


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
matplotlib.use('Agg')  # Set backend for non-interactive plotting






def run_deep_learning_model(file):
    # Implement your deep learning model logic here
    return "Deep Learning Prediction Result"

# def run_model(model_name, inputs):
#     if model_name == 'diabetes':
#         return "Diabetes Prediction Result"
#     elif model_name == 'cancer':
#         return "Cancer Prediction Result"
#     elif model_name == 'heart_attack':
#         return "Heart Attack Prediction Result"


@app.route('/')
def index():
    return render_template('index/index.html')

@app.route('/news')
def news():
    # Assuming you fetch data from an API
    news_articles = [{"title": "Health News 1", "description": "Details about news 1", "url": "#"}]
    return render_template('news.html', news_articles=news_articles)

# Replace with your actual Google Maps API key
GOOGLE_MAPS_API_KEY = 'YOUR_API_KEY'

@app.route('/hospitals', methods=['GET', 'POST'])
def hospitals():
    hospitals_data = []
    
    if request.method == 'POST':
        location = request.form['location']
        lat_lon = get_latitude_longitude(location)
        
        if lat_lon:
            latitude, longitude = lat_lon
            hospitals_data = fetch_nearby_hospitals(latitude, longitude)
    
    return render_template('location/hospitals.html', hospitals=hospitals_data)



@app.route('/search')
def search():
    query = request.args.get('query')
    # Web scraping for disease info using your scraping function
    disease_info = scrape_disease_info(query)
    return render_template('search_results.html', query=query, info=disease_info)














# Define deep learning models (for demonstration purposes)
deep_learning_models = ['Model A', 'Model B', 'Model C']  # Replace with actual model names

@app.route('/deep_learning', methods=['GET', 'POST'])
def deep_learning():
    if request.method == 'POST':
        file = request.files.get('image_file')
        if file:
            print("File uploaded:", file.filename)  # Debugging line
            prediction_result = run_deep_learning_model(file)  # Define this function
            return render_template('dl_results.html', prediction=prediction_result)

    return render_template('deep_learning.html')










# Define available models and their parameters
def load_model_config():
    config_path = os.path.join('front end', 'config', 'config.json')
    with open(config_path, 'r') as file:
        models_config = json.load(file)
    print("Loaded models config:", models_config)  # Add this line for debugging
    return models_config



@app.route('/basic_models', methods=['GET', 'POST'])
def basic_models_route():
    basic_models = load_model_config()  # Reload each time to ensure updated configuration

    if request.method == 'POST':
        selected_model = request.form.get('selected_model')
        
        if selected_model in basic_models:
            inputs = {param: request.form.get(param) for param in basic_models[selected_model]["features"]}
            # Redirect to prediction page with model and inputs
            return redirect(url_for('predict', model=selected_model, **inputs))
        else:
            flash("Selected model not found in configuration.", "danger")

    return render_template('basic_models.html', models=basic_models)


@app.route('/add_custom_model', methods=['GET', 'POST'])
def add_custom_model():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        target_column = request.form.get('target_column')
        file = request.files['file']

        if model_name and target_column and file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            try:
                data = pd.read_csv(file_path)
                model_path = train_and_save_model(data, target_column, model_name)
                flash(f"Model '{model_name}' created and saved at {model_path}", 'success')
                return redirect(url_for('model_visualizations', model_name=model_name, file_path= file_path))
            except Exception as e:
                flash(f"Error: {str(e)}", 'danger')
            return redirect(url_for('add_custom_model'))
        else:
            flash("Please provide all required fields.", 'warning')
        return render_template('add_custom_model.html')
    else:
        return render_template('add_custom_model.html')

@app.route('/model_visualizations/<model_name>/<file_path>')
def model_visualizations(model_name, file_path):
    config_data = load_model_config()
    if model_name not in config_data:
        flash("Model not found.", "danger")
        return redirect(url_for('basic_models_route'))

    # Load the data and generate visualizations
    # file_path = os.path.join(file_path)
    data = pd.read_csv(file_path)
    target_column = config_data[model_name]['target_column']
    fig_cm = create_visualizations(model_name, data, target_column)
    shap_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_shap_summary.png")
    corr_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_correlation_matrix.png")
    pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_pair_plot.png")
    # Convert figures to JSON for Plotly to render in HTML
    fig_cm_json = fig_cm.to_json()

    return render_template('model_visualizations.html', fig_cm=fig_cm_json,  shap_plot_path=shap_plot_path, corr = corr_plot_path, pair_plot_path=pair_plot_path)

























# Dictionary to store loaded models
models = {}

# Function to load available models from a specified directory
def load_model(models_dir, model_name):
    model_path = os.path.join(models_dir, model_name+".json")
    models[model_name] = xgb.XGBClassifier()
    models[model_name].load_model(model_path)


@app.route('/predict', methods=['GET'])
def predict():
    # Load models at startup

    # Get model choice from form
    model_choice = request.args.get('model')  # Changed to use query parameters
    
    # Check if the selected model is available
    if model_choice in load_model_config().keys():
        # Collect inputs from query parameters
        inputs = {key: request.args.get(key) for key in request.args if key != 'model'}
        # Convert inputs to appropriate types if necessary
        # Convert inputs to a 2D array format
        # input_values = [float(inputs[key]) for key in sorted(inputs.keys())]  # Ensure consistent order
        # input_array = np.array([input_values])  # Reshape to 2D array
        # Run prediction using the selected model
        prediction_result, force_plot_path = run_model(model_choice, inputs)
        load_model('Back end/chk_pts', model_choice)  # Ensure the model is loaded
        # predictions = models[model_choice].predict(input_array)  # Pass inputs as a list of dicts
        # Convert predictions to a list or DataFrame for display
        # prediction_result = predictions.tolist()
        return render_template('predict_results.html', 
                    result=prediction_result, 
                    model_name=model_choice, 
                    force_plot=force_plot_path)
    else:
        return "Model not found", 400









if __name__ == '__main__':
    app.run(debug=True)