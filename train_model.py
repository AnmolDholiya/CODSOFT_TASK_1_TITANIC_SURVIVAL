import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import model algorithms
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for Titanic Feature Engineering.
    Capsulates all preprocessing to ensure clean inference in Streamlit
    and prevent any form of data leakage during training.
    """
    def __init__(self):
        self.title_age_medians_ = {}
        self.fare_median_ = 14.45                      #Median Fare                         
        self.overall_age_median_ = 28.0              #Median Age
        self.embarked_mode_ = 'S'                        #Mode of Embarked
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        
        # 1. Title Extraction
        titles = X_df['Name'].apply(self._extract_title)
        X_df['Title'] = titles
        
        # 2. Compute medians for Age imputation based on Title
        self.title_age_medians_ = X_df.groupby('Title')['Age'].median().to_dict()
        self.overall_age_median_ = X_df['Age'].median()
        if pd.isna(self.overall_age_median_):
            self.overall_age_median_ = 28.0
            
        # 3. Compute median for Fare imputation
        self.fare_median_ = X_df['Fare'].median()
        if pd.isna(self.fare_median_):
            self.fare_median_ = 14.45
            
        # 4. Compute mode for Embarked
        if 'Embarked' in X_df.columns and not X_df['Embarked'].dropna().empty:
            self.embarked_mode_ = X_df['Embarked'].mode()[0]
        else:
            self.embarked_mode_ = 'S'
            
        return self
        
    def _extract_title(self, name):
        if not isinstance(name, str):
            return 'Mr'
        try:
            title = name.split(',')[1].split('.')[0].strip()
        except IndexError:
            return 'Mr'
            
        title_map = {
            'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master',
            'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Miss', 'Lady': 'Mrs',
            'Countess': 'Mrs', 'Dona': 'Mrs', 'Dr': 'Officer', 'Rev': 'Officer',
            'Col': 'Officer', 'Major': 'Officer', 'Capt': 'Officer',
            'Sir': 'Noble', 'Don': 'Noble', 'Jonkheer': 'Noble'
        }
        return title_map.get(title, 'Mr')
        
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        
        # Extract Title
        X_df['Title'] = X_df['Name'].apply(self._extract_title)
        
        # Impute Age based on Title medians
        def impute_age(row):
            if pd.isna(row['Age']):
                return self.title_age_medians_.get(row['Title'], self.overall_age_median_)
            return row['Age']
        
        # Apply age imputation row-by-row safely
        X_df['Age'] = X_df.apply(impute_age, axis=1)
        X_df['Age'] = X_df['Age'].fillna(self.overall_age_median_)
        
        # Impute Fare
        X_df['Fare'] = X_df['Fare'].fillna(self.fare_median_)
        
        # Impute Embarked
        X_df['Embarked'] = X_df['Embarked'].fillna(self.embarked_mode_)
        
        # Extract Cabin Deck
        def get_deck(cabin):
            if pd.isna(cabin) or not isinstance(cabin, str) or len(str(cabin).strip()) == 0:
                return 'U'
            return str(cabin).strip()[0].upper()
        X_df['Deck'] = X_df['Cabin'].apply(get_deck)
        
        # Feature Engineering: FamilySize and IsAlone
        X_df['SibSp'] = X_df['SibSp'].fillna(0).astype(int)
        X_df['Parch'] = X_df['Parch'].fillna(0).astype(int)
        X_df['FamilySize'] = X_df['SibSp'] + X_df['Parch'] + 1
        X_df['IsAlone'] = (X_df['FamilySize'] == 1).astype(int)
        X_df['FarePerPerson'] = X_df['Fare'] / X_df['FamilySize']
        
        # Feature Interaction
        X_df['AgeClass'] = X_df['Age'] * X_df['Pclass'].astype(float)
        
        # Explicit drop of columns not used in training
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        existing_drops = [c for c in cols_to_drop if c in X_df.columns]
        X_df = X_df.drop(columns=existing_drops)
        
        return X_df

def train_and_evaluate():
    # 1. Load the dataset
    data_path = "Titanic-Dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Split into features X and target y
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # ── Stratified 80/20 train-test split for honest evaluation ──────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test (held-out) size: {len(X_test)}")
    
    # 2. Define pipelines components
    categorical_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'Pclass']
    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'AgeClass']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # 3. Define candidate classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            random_seed=42,
            verbose=0
        )
    }
    
    # 4. Perform K-Fold cross validation on TRAIN SET ONLY to benchmark models
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_acc = 0.0
    best_model_name = None
    best_pipeline = None
    model_benchmarks = {}
    
    print("\n--- Model Training & Cross-Validation Benchmarks (on train set) ---")
    for name, clf in classifiers.items():
        # Create standard training pipeline
        pipeline = Pipeline(steps=[
            ('engineer', TitanicFeatureEngineer()),
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        try:
            # Cross validate on training data only (no test leakage)
            cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=skf,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                n_jobs=-1
            )
            
            mean_accuracy  = np.mean(cv_results['test_accuracy'])
            mean_precision = np.mean(cv_results['test_precision'])
            mean_recall    = np.mean(cv_results['test_recall'])
            mean_f1        = np.mean(cv_results['test_f1'])
            mean_auc       = np.mean(cv_results['test_roc_auc'])
        except Exception as e:
            print(f"  [{name}] CV failed: {e}")
            mean_accuracy = mean_precision = mean_recall = mean_f1 = mean_auc = float('nan')
        
        print(f"\nModel: {name}")
        print(f"  CV Accuracy:  {mean_accuracy:.4f}")
        print(f"  CV Precision: {mean_precision:.4f}")
        print(f"  CV Recall:    {mean_recall:.4f}")
        print(f"  CV F1 Score:  {mean_f1:.4f}")
        print(f"  CV ROC-AUC:   {mean_auc:.4f}")
        
        model_benchmarks[name] = {
            'accuracy':  mean_accuracy,
            'precision': mean_precision,
            'recall':    mean_recall,
            'f1':        mean_f1,
            'auc':       mean_auc
        }
        
        # Track the best model based on cross-validation accuracy
        if not np.isnan(mean_accuracy) and mean_accuracy > best_acc:
            best_acc = mean_accuracy
            best_model_name = name
            best_pipeline = pipeline
            
    print(f"\n>>> Best Model: {best_model_name} with {best_acc:.4f} cross-validation accuracy")
    
    # 5. Fit the best model on the FULL dataset (train + test combined)
    #    so the final saved model sees all available data.
    print(f"Fitting the final {best_model_name} model on the full dataset...")
    best_pipeline.fit(X, y)
    
    # 6. ── Honest held-out test set evaluation ───────────────────────────────
    #    We re-create and train an identical pipeline on X_train only,
    #    then evaluate on the untouched X_test.
    print("\n--- Held-Out Test Set Evaluation (honest generalization estimate) ---")
    eval_pipeline = Pipeline(steps=[
        ('engineer', TitanicFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('classifier', classifiers[best_model_name])
    ])
    eval_pipeline.fit(X_train, y_train)

    y_pred  = eval_pipeline.predict(X_test)
    y_proba = eval_pipeline.predict_proba(X_test)[:, 1]
    
    final_metrics = {
        'best_model': best_model_name,
        'accuracy':   accuracy_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'auc':        roc_auc_score(y_test, y_proba),
        'benchmarks': model_benchmarks,
        'eval_note':  'Metrics evaluated on a stratified 20% held-out test set (not training data).'
    }
    
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {final_metrics['auc']:.4f}")
    
    # 7. Extract Feature Importance for display in Streamlit
    print("\nExtracting feature importances...")
    preprocessor_fitted = best_pipeline.named_steps['preprocessor']
    
    cat_encoder = preprocessor_fitted.named_transformers_['cat']
    cat_features_transformed = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + cat_features_transformed
    
    fitted_clf = best_pipeline.named_steps['classifier']
    if hasattr(fitted_clf, 'feature_importances_'):
        importances = fitted_clf.feature_importances_
    elif hasattr(fitted_clf, 'get_feature_importance'):
        importances = fitted_clf.get_feature_importance()
    else:
        importances = np.zeros(len(all_feature_names))
        
    feature_imp_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    final_metrics['feature_importances'] = feature_imp_df.to_dict(orient='records')
    
    # 8. Save the final pipeline (trained on full data) and metrics
    model_filename   = 'titanic_model.joblib'
    metrics_filename = 'titanic_metrics.joblib'
    
    print(f"\nSaving final model pipeline to {model_filename}...")
    joblib.dump(best_pipeline, model_filename)
    
    print(f"Saving final model metrics to {metrics_filename}...")
    joblib.dump(final_metrics, metrics_filename)
    
    print("\nModel Core Development Complete! Ready to launch Streamlit Dashboard.")

if __name__ == "__main__":
    train_and_evaluate()
