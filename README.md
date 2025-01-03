# credit-card-fraud-detection
# Overview:
## Project Structure
Credit_Card_Fraud_Detection/ <br>
├── app.py <br>
├── flask_app.py <br>
├── config/ <br>
│ └── config.yml <br>
├── data/  <br>
│ ├── train.csv <br>
│ ├── test.csv  <br>
│ └── README.md  <br>
├── notebooks/ <br>
│ └── EDA.ipynb  <br>
├── pycache/  <br>
├── README.md  <br>
├── requirements.txt  <br>
├── saves/  <br>
│ ├── models/  <br>
│ └── visualizations/  <br>
├── src/  <br>
│ ├── base_lines.py  <br>
│ ├── credit_fraud_utils_data.py  <br>
│ ├── credit_fraud_utils_eval.py  <br>
│ ├── init.py  <br>
│ ├── train.py  <br>
│ ├── utils.py  <br>
│ └── predict.py  <br>
├── templates/   <br>
│ ├── index.html  <br>
│ └── result.html  <br>

# Install Required Packages
pip install -r requirements.txt
# Train Logistic Regression Model with Cross-Validation and Save
python src/train.py --model logistic_regression --cross-val --save-model
# Available models :
logistic_regression <br>
xgboost <br>
random_forest <br>
neural_network <br>
voting_classifier <br>
