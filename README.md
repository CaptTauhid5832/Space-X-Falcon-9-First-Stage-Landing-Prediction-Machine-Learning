# Space-X-Falcon-9-First-Stage-Landing-Prediction-Machine-Learning
SpaceX Falcon 9 First Stage Landing Prediction
This project predicts whether the Falcon 9 first stage will land successfully. This is a critical task because SpaceX saves millions of dollars by reusing the first stage of the rocket.

Project Overview
The goal is to determine the probability of a successful landing using machine learning. The project involves:

Data Collection: Gathering data via the SpaceX API and web scraping.

EDA: Analyzing trends in launch sites, payload mass, and orbits.

Interactive Mapping: Using Folium to visualize launch sites and landing success rates.

Dashboarding: Creating an interactive Plotly Dash app to filter data by site and payload.

Machine Learning: Training and comparing Logistic Regression, SVM, Decision Tree, and KNN models.

Technologies Used
Python 3.13

Pandas & NumPy (Data Manipulation)

Seaborn & Matplotlib (Static Visualization)

Folium (Geospatial Mapping)

Plotly Dash (Interactive Dashboard)

Scikit-Learn (Machine Learning & Preprocessing)

Key Findings
Best Model: Logistic Regression, SVM, and KNN all performed with a test accuracy of approximately 83.3%.

Launch Trends: Success rates have improved significantly over time as SpaceX refined their landing technology.

Payload Impact: Heavier payloads and specific orbits (like GTO) show different success correlations compared to lighter LEO missions.

How to Run
1. Clone the repository
Bash
git clone https://github.com/CaptTauhid5832/spacex-predict.git
cd spacex-predict
2. Install dependencies
Bash
pip install pandas numpy seaborn matplotlib folium plotly dash scikit-learn requests
3. Run the Dashboard
Bash
python spacex_dash.py
4. View the Analysis
Open the Jupyter Notebooks in the notebooks/ folder to see the step-by-step data processing and model training.

File Structure
spacex_dash.py: The Plotly Dash application.

notebooks/EDA_Visualization.ipynb: Exploratory data analysis using Seaborn.

notebooks/Machine_Learning_Prediction.ipynb: Model training and hyperparameter tuning.

notebooks/Folium_Maps.ipynb: Geographical analysis of launch sites.
