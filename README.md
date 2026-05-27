# 🚢 Titanic Survival Prediction Web App & ML Model

A state-of-the-art machine learning project utilizing the classic Titanic passenger dataset to build a highly accurate predictive model, paired with a gorgeous, high-end interactive Streamlit web application.

## ✨ Features
* **Machine Learning Pipeline (`train_model.py`)**:
  * Advanced feature extraction (e.g. Title, Cabin Deck, Family Size, Alone status, Fare-per-Person, and Class-Age interactions).
  * Group-median age imputation based on the passenger's exact title group.
  * Cross-validated benchmark evaluations of multiple classifiers (Random Forest, XGBoost).
  * Optimized final pipeline (`titanic_model.joblib`) serialized for inference.
* **Premium Interactive Web App (`app.py`)**:
  * **Interactive Survival Predictor**: Real-time survival chance update with styled interactive Plotly gauges based on user input.
  * **Exploratory Data Analysis (EDA)**: Gorgeous custom-color dark mode visualizations outlining survival rates by Gender, Class, Age demographics, and Fare distributions.
  * **Model Performance tab**: Explains feature weightings (Feature Importances) and reports accuracy, precision, recall, and ROC-AUC score.
  * **Batch prediction sandbox**: Upload custom CSV manifest spreadsheets to instantly predict survival rates in bulk and download the results.

---

## 📈 Final Model Benchmarks

| Metric | Random Forest | XGBoost (Selected) |
| :--- | :--- | :--- |
| **5-Fold CV Accuracy** | 83.61% | **84.40%** |
| **5-Fold CV Precision** | 83.03% | **82.92%** |
| **5-Fold CV ROC-AUC** | 87.88% | **88.52%** |
| **Final Training Set Accuracy** | 88.33% | **91.36%** |

---

## 🛠️ Project Structure
```
├── Titanic-Dataset.csv       # Original dataset
├── train_model.py            # Preprocessing and model training script
├── app.py                    # Streamlit web dashboard application
├── titanic_model.joblib      # Serialized scikit-learn pipeline (model)
├── titanic_metrics.joblib    # Serialized model metrics & importances
└── README.md                 # Project documentation
```

---

## 🚀 How to Run the Web Application

1. **Install Dependencies**
   Make sure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit plotly joblib pillow
   ```

2. **Train the Model (Optional)**
   The serialized model pipelines are already created! If you wish to retrain or modify the algorithms, run:
   ```bash
   python train_model.py
   ```

3. **Launch the Streamlit Web Server**
   Start the local web application:
   ```bash
   streamlit run app.py
   ```
   A browser window will open automatically displaying the application at `http://localhost:8501`.
