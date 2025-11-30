from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib
import pandas as pd
import numpy as np

class ChurnModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def prepare_data(self, features, target_col='churned', test_size=0.2):
        """Split data into train and test sets"""
        X = features.drop([target_col, 'customer_id'], axis=1, errors='ignore')
        y = features[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['Logistic Regression'] = model
        print("✓ Trained Logistic Regression")
        return model
    
    def train_decision_tree(self, X_train, y_train):
        """Train Decision Tree model"""
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        self.models['Decision Tree'] = model
        print("✓ Trained Decision Tree")
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        model.fit(X_train, y_train)
        self.models['Random Forest'] = model
        print("✓ Trained Random Forest")
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
        model.fit(X_train, y_train)
        self.models['XGBoost'] = model
        print("✓ Trained XGBoost")
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        model = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
        model.fit(X_train, y_train)
        self.models['LightGBM'] = model
        print("✓ Trained LightGBM")
        return model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network (ANN) model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        self.models['Neural Network'] = model
        print("✓ Trained Neural Network")
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        if model_name == 'Neural Network':
            y_pred_proba = model.predict(X_test).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models and return results"""
        print("\n" + "="*50)
        print("TRAINING ALL MODELS")
        print("="*50)
        
        self.train_logistic_regression(X_train, y_train)
        self.train_decision_tree(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_neural_network(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results_df = pd.DataFrame()
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return self.models, self.results
    
    def save_models(self, path='models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'Neural Network':
                model.save(f"{path}{name.replace(' ', '_').lower()}.h5")
            else:
                joblib.dump(model, f"{path}{name.replace(' ', '_').lower()}.pkl")
        
        print(f"✓ Saved {len(self.models)} models to {path}")

class CustomerSegmentation:
    """Perform customer segmentation using clustering"""
    
    def __init__(self, features):
        self.features = features
        
    def kmeans_clustering(self, n_clusters=4):
        """Perform K-Means clustering"""
        cluster_features = ['tenure_months', 'monthly_charges', 'total_charges', 
                          'purchase_frequency', 'engagement_score']
        
        available_cols = [col for col in cluster_features if col in self.features.columns]
        X = self.features[available_cols].fillna(0)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.features['cluster'] = clusters
        
        print(f"✓ K-Means clustering complete: {n_clusters} segments")
        print(self.features['cluster'].value_counts().sort_index())
        
        return clusters, kmeans
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering for anomaly detection"""
        cluster_features = ['tenure_months', 'monthly_charges', 'purchase_frequency']
        available_cols = [col for col in cluster_features if col in self.features.columns]
        X = self.features[available_cols].fillna(0)
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        self.features['dbscan_cluster'] = clusters
        
        n_anomalies = (clusters == -1).sum()
        print(f"✓ DBSCAN clustering complete")
        print(f"  Detected {n_anomalies} anomalies")
        
        return clusters, dbscan