# === Import necessary libraries ===
from flask import Flask, render_template, request  # Flask for web framework and HTML rendering
import pandas as pd  # For data manipulation and CSV reading
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # High-level plotting API built on top of matplotlib
from sklearn.linear_model import LinearRegression  # Linear regression model for predictions
import numpy as np  # For numerical operations
import joblib  # For saving/loading trained ML models
import os  # File and directory operations

# === Initialize Flask app ===
app = Flask(__name__)

# === Load the dataset from CSV file ===
data = pd.read_csv('a.csv')

# === Convert categorical variables to numerical codes ===
# Save the original categories for dropdowns or decoding
crop_type_mapping = data['CropType'].astype('category').cat.categories.tolist()
region_mapping = data['Region'].astype('category').cat.categories.tolist()
season_mapping = data['Season'].astype('category').cat.categories.tolist()
soil_type_mapping = data['SoilType'].astype('category').cat.categories.tolist()

# Replace categorical text with numeric codes for ML model training
data['CropType'] = data['CropType'].astype('category').cat.codes
data['Region'] = data['Region'].astype('category').cat.codes
data['Season'] = data['Season'].astype('category').cat.codes
data['SoilType'] = data['SoilType'].astype('category').cat.codes

# === Define features (X) and target variable (y) ===
X = data[['CropType', 'Region', 'Season', 'SoilType', 'Rainfall (mm)', 'Temperature (°C)',
          'FertilizerUsed (kg)', 'PesticidesUsed (kg)']]
y = data['InventoryLevel (kg)']

# === Train Linear Regression Model ===
model = LinearRegression()
model.fit(X, y)

# === Save trained model to disk if not already saved ===
model_filename = 'inventory_prediction_model.pkl'
if not os.path.exists(model_filename):
    joblib.dump(model, model_filename)  # Save model using joblib

# === Generate plots and predictions based on user-selected category ===
def generate_plots(category):
    prediction_data = X.copy()
    prediction_data[category] = prediction_data[category] + 1  # Modify the selected feature
    prediction = model.predict(prediction_data).tolist()  # Predict and convert to list

    plot_images = []
    plot_descriptions = []

    # Plot 1: Inventory vs Rainfall
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['Rainfall (mm)'], y=data['InventoryLevel (kg)'])
    plt.title('Inventory vs Rainfall')
    plot_path = 'static/images/inventory_vs_rainfall.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)
    plot_descriptions.append("This plot shows how rainfall impacts inventory levels.")

    # Plot 2: Inventory vs Temperature
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['Temperature (°C)'], y=data['InventoryLevel (kg)'])
    plt.title('Inventory vs Temperature')
    plot_path = 'static/images/inventory_vs_temp.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)
    plot_descriptions.append("This plot shows the relationship between temperature and inventory.")

    # Plot 3: Demand Forecast vs Inventory
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=data['DemandForecast (kg)'], y=data['InventoryLevel (kg)'])
    plt.title('Demand Forecast vs Inventory')
    plot_path = 'static/images/demand_vs_inventory.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)
    plot_descriptions.append("This line graph shows how inventory corresponds with forecasted demand.")

    # Plot 4: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plot_path = 'static/images/corr_heatmap.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)
    plot_descriptions.append("This heatmap shows correlation between all numeric features.")

    # Plot 5: Predicted vs Actual Inventory
    plt.figure(figsize=(8, 6))
    plt.scatter(data['InventoryLevel (kg)'], prediction)
    plt.plot([min(data['InventoryLevel (kg)']), max(data['InventoryLevel (kg)'])],
             [min(prediction), max(prediction)], color='red')
    plt.title('Predicted vs Actual Inventory Level')
    plot_path = 'static/images/predicted_vs_actual.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)
    plot_descriptions.append("This plot shows the accuracy of model predictions.")

    return plot_images, plot_descriptions, prediction

# === Main route (index) ===
@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    plot_images = []
    plot_descriptions = []
    prediction = None

    if request.method == 'POST':
        category = request.form.get('category')
        plot_images, plot_descriptions, prediction = generate_plots(category)

    return render_template('main_page.html', plot_images=plot_images,
                           plot_descriptions=plot_descriptions,
                           category=category, prediction=prediction)

# === Alternative prediction route ===
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    category = None
    plot_images = []
    plot_descriptions = []
    prediction = None

    if request.method == 'POST':
        category = request.form.get('category')
        plot_images, plot_descriptions, prediction = generate_plots(category)

    return render_template('index.html', plot_images=plot_images,
                           plot_descriptions=plot_descriptions,
                           category=category, prediction=prediction)

# === Route for stock management visualization ===
def generate_stock_management_plots():
    model = joblib.load(model_filename)  # Load trained model
    prediction = model.predict(X)  # Predict on entire dataset

    plot_images = []

    # Inventory vs Rainfall
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['Rainfall (mm)'], y=data['InventoryLevel (kg)'])
    plt.title('Inventory vs Rainfall')
    plot_path = 'static/images/inventory_vs_rainfall.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)

    # Inventory vs Temperature
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['Temperature (°C)'], y=data['InventoryLevel (kg)'])
    plt.title('Inventory vs Temperature')
    plot_path = 'static/images/inventory_vs_temp.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)

    # Predicted vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(data['InventoryLevel (kg)'], prediction)
    plt.plot([min(data['InventoryLevel (kg)']), max(data['InventoryLevel (kg)'])],
             [min(prediction), max(prediction)], color='red')
    plt.title('Predicted vs Actual Inventory Level')
    plot_path = 'static/images/predicted_vs_actual.png'
    plt.savefig(plot_path)
    plot_images.append(plot_path)

    # Generate insights
    insights = []
    low_inventory_items = data[data['InventoryLevel (kg)'] < data['DemandForecast (kg)']]
    if not low_inventory_items.empty:
        for _, row in low_inventory_items.iterrows():
            item_name = row['CropType']
            inventory_level = row['InventoryLevel (kg)']
            demand_forecast = row['DemandForecast (kg)']
            insights.append(f"Item {item_name} (Inventory Level: {inventory_level} kg) is running low. Demand Forecast: {demand_forecast} kg.")
    else:
        insights.append("All items have sufficient stock compared to forecasted demand.")

    return plot_images, insights

@app.route('/stock_management', methods=['GET'])
def stock_management():
    plot_images, insights = generate_stock_management_plots()
    return render_template('stock_management.html', plot_images=plot_images, insights=insights)

# === Custom prediction route (user inputs) ===
@app.route('/custom_prediction', methods=['GET', 'POST'])
def custom_prediction():
    prediction = None
    demand = None
    stock_status = None
    error_message = None
    explanation = None

    # Dropdown values for the form
    crop_types = crop_type_mapping
    regions = region_mapping
    seasons = season_mapping
    soil_types = soil_type_mapping

    if request.method == 'POST':
        try:
            crop_type = request.form.get('CropType')
            region = request.form.get('Region')
            season = request.form.get('Season')
            soil_type = request.form.get('SoilType')
            rainfall = float(request.form.get('Rainfall'))
            temperature = float(request.form.get('Temperature'))
            fertilizer_used = float(request.form.get('FertilizerUsed'))
            pesticides_used = float(request.form.get('PesticidesUsed'))

            # Convert inputs to model-readable format
            crop_type_code = crop_types.index(crop_type)
            region_code = regions.index(region)
            season_code = seasons.index(season)
            soil_type_code = soil_types.index(soil_type)

            # Prepare input for inventory model
            prediction_data = np.array([[crop_type_code, region_code, season_code, soil_type_code,
                                         rainfall, temperature, fertilizer_used, pesticides_used]])
            prediction = model.predict(prediction_data)[0]

            # === FIX: Proper Demand Calculation ===
            demand = (rainfall * 1.5) + (temperature * 2)

            # === Debugging logs ===
            print(f"[DEBUG] Inputs → Rainfall: {rainfall}, Temperature: {temperature}")
            print(f"[DEBUG] Calculated Demand: {demand}")
            print(f"[DEBUG] Predicted Inventory: {prediction}")

            # Determine stock status
            if prediction < demand:
                stock_status = "Understock"
            elif prediction > demand:
                stock_status = "Overstock"
            else:
                stock_status = "Optimal Stock"

            explanation = generate_explanation(crop_type, region, season, soil_type,
                                               rainfall, temperature, fertilizer_used, pesticides_used)

        except Exception as e:
            error_message = f"An error occurred: {e}"

    return render_template('custom_prediction.html', prediction=prediction, demand=demand,
                           stock_status=stock_status, explanation=explanation,
                           error_message=error_message, crop_types=crop_types,
                           regions=regions, seasons=seasons, soil_types=soil_types)

# === Generate scientific explanation for inputs ===
def generate_explanation(crop_type, region, season, soil_type, rainfall, temperature, fertilizer_used, pesticides_used):
    explanation = f"Based on your inputs, the predicted inventory level is calculated as follows:\n\n"
    explanation += f"1. **Crop Type ({crop_type})** affects input requirements.\n"
    explanation += f"2. **Region ({region})** impacts weather, pests, and soil fertility.\n"
    explanation += f"3. **Season ({season})** influences water needs and crop cycles.\n"
    explanation += f"4. **Soil Type ({soil_type})** determines water retention and nutrient levels.\n"
    explanation += f"5. **Rainfall ({rainfall} mm)** affects irrigation needs.\n"
    explanation += f"6. **Temperature ({temperature} °C)** impacts crop metabolism.\n"
    explanation += f"7. **Fertilizer Used ({fertilizer_used} kg)** boosts growth if applied properly.\n"
    explanation += f"8. **Pesticides Used ({pesticides_used} kg)** protect crops from yield-reducing pests.\n"
    return explanation

# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)
