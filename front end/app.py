from flask import Flask, render_template, request, redirect, url_for,flash,jsonify
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







@app.route('/')
def index():
    return render_template('index/index.html')

@app.route('/news')
def news():
    # Assuming you fetch data from an API
    news_articles = [{"title": "Health News 1", "description": "Details about news 1", "url": "#"}]
    return render_template('news.html', news_articles=news_articles)

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

@app.route('/models')
def models():
    return render_template('index/predict.html')

# Define available models and their parameters

def preprocess_data(data,target_column):
    data.dropna()
    # Encode categorical features
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes
    # Split dataset into features (X) and target (y)
    y = data[target_column]


    # Get unique classes and their counts
    class_counts = y.value_counts()

    # Identify minimum and maximum class counts
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()

    # Balance classes by reducing larger classes if necessary
    balanced_data = []

    for cls in class_counts.index:
        cls_data = data[data[target_column] == cls]

        if len(cls_data) > min_class_count * 2:
            # Downsample larger classes to double the size of the minimum class
            cls_data = resample(cls_data, replace=False, n_samples=min_class_count * 2, random_state=42)

        balanced_data.append(cls_data)
        print(len(balanced_data))

    # Combine balanced classes into a single DataFrame
    balanced_data = pd.concat(balanced_data)
    return balanced_data

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

    return render_template('basic_model/basic_models.html', models=basic_models)


@app.route('/add_custom_model', methods=['GET', 'POST'])
def add_custom_model():
    if request.method == 'POST':
        model_name = request.form.get('model_name')
        target_column = request.form.get('target_column')
        file = request.files['file']

        if model_name and target_column and file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(UPLOAD_FOLDER+ file.filename)
            try :
                file.save(file_path)
                data = pd.read_csv(file_path)
                balanced_data= preprocess_data(data, target_column)
                model_path = train_and_save_model(balanced_data, target_column, model_name)
                flash(f"Model '{model_name}' created and saved at {model_path}", 'success')
                return redirect(url_for('model_visualizations', model_name=model_name, file_path= file_path))
            except Exception as e:
                error_message = str(e) or "An unexpected error occurred."
                return render_template('basic_model/add_custom_model.html', 
                             error_message=error_message)
        else:
            flash("Please provide all required fields.", 'warning')
            return render_template('basic_model/add_custom_model.html')
    elif request.method == 'GET':
        return render_template('basic_model/add_custom_model.html')

@app.route('/model_visualizations/<model_name>/<file_path>')
def model_visualizations(model_name, file_path):
    config_data = load_model_config()
    if model_name not in config_data:
        flash("Model not found.", "danger")
        return redirect(url_for('basic_models_route'))

    # Load the data and generate visualizations
    data = pd.read_csv(file_path)
    target_column = config_data[model_name]['target_column']
    data = preprocess_data(data, target_column)
    
    # Generate visualizations and insights
    fig_cm, report_summary = create_visualizations(model_name, data, target_column)
    
    shap_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_shap_summary.png")
    corr_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_correlation_matrix.png")
    pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_pair_plot.png")
    
    # Convert figures to JSON for Plotly to render in HTML
    fig_cm_json = fig_cm.to_json()

    return render_template('basic_model/model_visualizations.html', 
                           fig_cm=fig_cm_json,
                           shap_plot_path=shap_plot_path,
                           corr=corr_plot_path,
                           pair_plot_path=pair_plot_path,
                           report_summary=report_summary)  # Pass report summary to template

# Dictionary to store loaded models
models = {}

# Function to load available models from a specified directory
def load_model(models_dir, model_name):
    model_path = os.path.join(models_dir, model_name+".json")
    models[model_name] = xgb.XGBClassifier()
    models[model_name].load_model(model_path)


@app.route('/predict', methods=['GET'])
def predict():
    # Get model choice from form
    model_choice = request.args.get('model')  # Changed to use query parameters
    
    # Check if the selected model is available
    if model_choice in load_model_config().keys():
        # Collect inputs from query parameters
        inputs = {key: request.args.get(key) for key in request.args if key != 'model'}

        prediction_result, force_plot_path = run_model(model_choice, inputs)
        load_model('Back end/chk_pts', model_choice)  # Ensure the model is loaded
        shap_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_choice}_shap_summary.png")
        corr_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_choice}_correlation_matrix.png")
        pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_choice}_pair_plot.png")
    
        return render_template('basic_model/predict_results.html', 
                    result=prediction_result, 
                    model_name=model_choice, 
                    force_plot=force_plot_path,
                           shap_plot_path=shap_plot_path,
                           corr=corr_plot_path,
                           pair_plot_path=pair_plot_path)
    else:
        return "Model not found", 400


























# Define deep learning models (for demonstration purposes)
# Load available models from config file
def load_model_config2(config_path='front end\config\config2.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['models']
    return {}

# Save model configuration to the config file
def save_model_config2(models, config_path='front end\config\config2.json'):
    with open(config_path, 'w') as f:
        json.dump({"models": models}, f, indent=4)

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    if 'model_name' not in request.form or 'weights_file' not in request.files:
        return jsonify({"error": "Model name and weights file are required."}), 400
    
    model_name = request.form['model_name']
    weights_file = request.files['weights_file']

    if weights_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Save the uploaded weights file
    weights_path = os.path.join('weights', weights_file.filename)  # Ensure you have a 'weights' directory
    weights_file.save(weights_path)

    # Load existing models
    models = load_model_config2()
    
    # Update the model's weight path
    models[model_name] = weights_path
    
    # Save updated models back to config file
    save_model_config2(models)

    return jsonify({"message": f"Weights for {model_name} uploaded successfully."}), 200

@app.route('/deep_learning', methods=['GET', 'POST'])
def deep_learning():
    if request.method == 'POST':
        file = request.files.get('image_file')
        model_name = request.form.get('model_name')  # Get selected model name from form
        
        if file and model_name:
            print("File uploaded:", file.filename)  # Debugging line
            
            # Run deep learning prediction with the selected model
            prediction_result = run_deep_learning_model(file, model_name)
            return render_template('dl_results.html', prediction=prediction_result)

    # Load available models for selection in the HTML form
    models = load_model_config2()
    
    return render_template('dl_model/deep_learning.html', models=models)







if __name__ == '__main__':
    app.run(debug=True)