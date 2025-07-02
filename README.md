# S&P500-Stock-Prediction
Code that predicts S&P500 percentage change using machine learning techniques. It utilizes historical finance data from sources from the Yahoo finance to train a Gradient Boosting model.

# Tools and Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- yFinance
- datetime
- Matplotlib
- Seaborn

# Features
- Data Collection: Fetches Finance data using the yFinance, including historical data from various stocks for stronger model training
- Data Processing: Converts and preprocesses data into a format suitable for machine learning models. Includes datetime conversion, feature engineering, and encoding categorical variables.
- Model Training: Utilizes XGBoost to train on historical Finance data. Evaluates model performance using metrics such as accuracy, precision, recall, and R^2-score.
- Prediction: Makes predictions on the future value of the S&P 500 using a variable amount of historical data
- Visualization: Visualizes data trends and model insights using Matplotlib and Seaborn.
- 
# How to Use
Setup Environment: Install required libraries using pip install -r requirements
