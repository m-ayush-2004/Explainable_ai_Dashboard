import os
import json
import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Configurations
CONFIG_FILE_PATH = 'front end/config/config.json'
MODELS_DIR = 'Back end/chk_pts/'
SHAP_VALUES_DIR = 'back end/shap_values/'
SHAP_EXPLAINER_DIR = 'back end/shap_explainer/'
SHAP_PLOT_DIR = 'front end/static/shap_plots/'

MAX_SHAP_SAMPLES = 200  # Set maximum number of samples for SHAP

def update_config_file(model_name, target_column, feature_columns):
    config_data = {}
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as file:
            config_data = json.load(file)
    config_data[model_name] = {'target_column': target_column, 'features': feature_columns}
    with open(CONFIG_FILE_PATH, 'w') as file:
        json.dump(config_data, file, indent=4)

def train_and_save_model(data, target_column, model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.json")
    shap_values_path = os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap_values.npz")
    shap_explainer_path = os.path.join(SHAP_EXPLAINER_DIR, f"{model_name}_explainer.pkl")  # Path for saving the explainer
    # Check if model and SHAP values already exist
    if os.path.exists(model_path) and os.path.exists(shap_values_path):
        print(f"Model and SHAP values for '{model_name}' already exist. Skipping training.")
        return model_path

    # Split dataset into features (X) and target (y)
    x = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical features
    for col in x.select_dtypes(include=['object']).columns:
        x[col] = x[col].astype('category').cat.codes

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(x, y)

    # Save model
    model.save_model(model_path)

    # Generate and save SHAP values with sampling
    shap_values, x_sampled = generate_shap_values(model, x)
    np.savez_compressed(shap_values_path, shap_values=shap_values.values, base_values=shap_values.base_values, features=x_sampled)

    # Save the explainer (you can save it as a pickle)
    with open(shap_explainer_path, 'wb') as f:
        pickle.dump(shap.Explainer(model), f)
    # Update config file
    update_config_file(model_name, target_column, list(x.columns))
    return model_path

def generate_shap_values(model, x):
    # Limit dataset to 3,000 samples for SHAP analysis
    if len(x) > MAX_SHAP_SAMPLES:
        x_sampled = x.sample(n=MAX_SHAP_SAMPLES, random_state=42)
    else:
        x_sampled = x

    # Run SHAP analysis on the sampled data
    explainer = shap.Explainer(model.predict, x_sampled)
    shap_values = explainer(x_sampled)
    return shap_values, x_sampled

def create_visualizations(model_name, data, target_column):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.json")
    shap_values_path = os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap_values.npz")
    
    
    # Encode categorical features
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype('category').cat.codes
    
    x = data.drop(columns=[target_column])
    # Load the saved model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Check if SHAP values already exist
    if not os.path.exists(shap_values_path):
        print(f"SHAP values for '{model_name}' do not exist. Please run training first.")
        return None
    else:
        # Load SHAP values and the feature sample
        shap_data = np.load(shap_values_path, allow_pickle=True)
        
        shap_values = shap_data['shap_values']
        base_values = shap_data['base_values']
        x_sampled = shap_data['features']

        # Generate SHAP summary plot with the sampled features 
        plt.figure()
        shap.violin_plot(shap_values, features=pd.DataFrame(x_sampled), show=False)
        plt.savefig(os.path.join(SHAP_PLOT_DIR, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    # Generate confusion matrix
    y_pred = model.predict(x)
    conf_matrix = confusion_matrix(data[target_column], y_pred)
    fig_cm = px.imshow(conf_matrix,
                        labels={'x': 'Predicted', 'y': 'Actual'},
                        color_continuous_scale='Blues')

    # Check if pair plot already exists; if not, create it.
    pair_plot_path = os.path.join(SHAP_PLOT_DIR, f"{model_name}_pair_plot.png")
    if not os.path.exists(pair_plot_path):
        print("Generating pair plot...")
        plt.figure(figsize=(10, 8))
        sns.pairplot(data.sample(n=min(1000,len(data))), hue=target_column)  # Sample to reduce time
        plt.savefig(pair_plot_path, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        print(f"Pair plot for '{model_name}' already exists. Skipping generation.")

    # Calculate and plot the correlation matrix using Seaborn
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8}, xticklabels=data.columns, yticklabels=data.columns)
    
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(SHAP_PLOT_DIR, f"{model_name}_correlation_matrix.png"), bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    return fig_cm  # Returning Plotly figures

def run_model(selected_model, inputs):
    model_path = os.path.join(MODELS_DIR, f"{selected_model}.json")
    shap_explainer_path = os.path.join(SHAP_EXPLAINER_DIR, f"{selected_model}_explainer.pkl")  # Path for SHAP explainer

    # Load the model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Prepare input data for prediction
    input_array = np.array([list(inputs.values())], dtype=float)  # Reshape inputs for prediction

    # Get predictions
    predictions = model.predict(input_array)

    # Generate SHAP force plot and save it as an image file
    force_plot_path = os.path.join(SHAP_PLOT_DIR, f"{selected_model}_force_plot.png")
    
    # Load existing SHAP explainer
    with open(shap_explainer_path, 'rb') as f:
        explainer = pickle.load(f)

    # Calculate SHAP values for the input using loaded explainer
    input_shap_values = explainer(input_array)

    # Create a force plot using matplotlib
    shap.force_plot(explainer.expected_value, input_shap_values.values, input_array, matplotlib=True)
    
    # Save the figure
    plt.savefig(force_plot_path)
    plt.close()  # Close the figure to free memory

    return predictions[0], force_plot_path  # Return prediction and force plot HTML