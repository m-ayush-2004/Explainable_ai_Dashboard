from flask import Flask, render_template, request, redirect, url_for
from disease_scraper import scrape_disease_info
app = Flask(__name__)



def run_deep_learning_model(file):
    # Implement your deep learning model logic here
    return "Deep Learning Prediction Result"

def run_model(model_name, inputs):
    if model_name == 'diabetes':
        return "Diabetes Prediction Result"
    elif model_name == 'cancer':
        return "Cancer Prediction Result"
    elif model_name == 'heart_attack':
        return "Heart Attack Prediction Result"








@app.route('/')
def index():
    return render_template('index.html')

@app.route('/news')
def news():
    # Assuming you fetch data from an API
    news_articles = [{"title": "Health News 1", "description": "Details about news 1", "url": "#"}]
    return render_template('news.html', news_articles=news_articles)

@app.route('/hospitals')
def hospitals():
    hospitals_data = [{"name": "Hospital 1", "address": "1234 Street", "phone": "123-456-7890"}]
    return render_template('hospitals.html', hospitals=hospitals_data)

@app.route('/search')
def search():
    query = request.args.get('query')
    # Web scraping for disease info using your scraping function
    disease_info = scrape_disease_info(query)
    return render_template('search_results.html', query=query, info=disease_info)

# Define available models and their parameters
basic_models = {
    'diabetes': ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
       'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'],
    'heart_attack': ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
       'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'],
    'stroke_analysis': ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke'],
    'tissue_tumor':['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32']
}

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

@app.route('/basic_models', methods=['GET', 'POST'])
def basic_models_route():
    if request.method == 'POST':
        selected_model = request.form.get('selected_model')
        inputs = {param: request.form.get(param) for param in basic_models[selected_model]}
        prediction_result = run_model(selected_model, inputs)  # Define this function
        return render_template('basic_results.html', prediction=prediction_result)

    return render_template('basic_models.html', models=basic_models)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html')




if __name__ == '__main__':
    app.run(debug=True)