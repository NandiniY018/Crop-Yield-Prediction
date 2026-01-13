## Crop Yield Prediction
  Crop Yield Prediction is a machine learning-based web application designed to estimate agricultural productivity.  It predicts crop yield using key environmental and agricultural factors such as 
  " rainfall, temperature, soil type, fertilizer usage, and cultivated area " .

## System Description
  The system uses machine learning models trained on **historical agricultural data**.  
  It identifies relationships between environmental factors and crop yield.  
  Users can input values such as rainfall, temperature, soil type, and fertilizer amount.  
  The trained model processes this input and displays the **predicted yield** instantly.  
  The web interface is user-friendly, interactive, and supports data visualization.  

## Technologies Used
  **Programming Language:** Python  
  **Web Framework:** Flask
  **Machine Learning Libraries:** Scikit-learn, Pandas, NumPy  
  **Database:** SQLite  
  **Frontend Technologies:** HTML, CSS, JavaScript  
  **Visualization Tools:** Matplotlib, Seaborn (for data analysis)  
  **Version Control:** Git and GitHub  
  **Deployment (optional):** AWS 

## Machine Learning Model
  Algorithms used include: Linear Regression, Decision Tree, Regression, Random Forest Regression 
  After evaluation, **Random Forest** and **XGBoost** provided the best performance with high accuracy and low error rates.  
  **Data preprocessing** is done using Pandas and NumPy to clean, normalize, and prepare the dataset.  
  The model is trained on data including: Crop Name, Soil Type, Temperature, Rainfall, Fertilizer Quantity, Actual Yield  
