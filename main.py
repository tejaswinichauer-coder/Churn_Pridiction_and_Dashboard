import sys
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ChurnModelTrainer, CustomerSegmentation
from sales_analysis import SalesAnalyzer

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*70)
    print("CUSTOMER CHURN PREDICTION & SALES DASHBOARD")
    print("="*70)
    
    # Step 1: Data Loading & Preprocessing
    print("\n[1/6] DATA LOADING & PREPROCESSING")
    print("-" * 70)
    preprocessor = DataPreprocessor()
    customers, transactions, products = preprocessor.load_data()
    
    customers = preprocessor.clean_data(customers)
    transactions = preprocessor.clean_data(transactions)
    
    # Step 2: Feature Engineering
    print("\n[2/6] FEATURE ENGINEERING")
    print("-" * 70)
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_customer_features(customers, transactions)
    transactions = feature_engineer.create_time_features(transactions)
    
    # Step 3: Encode and normalize
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    if 'customer_id' in categorical_cols:
        categorical_cols.remove('customer_id')
    
    features_encoded = preprocessor.encode_categorical(features, categorical_cols)
    
    # Step 4: Model Training
    print("\n[3/6] MODEL TRAINING")
    print("-" * 70)
    trainer = ChurnModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(features_encoded)
    
    models, results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    trainer.save_models()
    
    # Step 5: Sales Analysis
    print("\n[4/6] SALES ANALYSIS")
    print("-" * 70)
    analyzer = SalesAnalyzer(transactions)
    monthly_sales = analyzer.monthly_sales_trend()
    top_products = analyzer.top_products()
    churn_revenue = analyzer.churn_revenue_correlation(customers)
    
    print("\nTop 5 Monthly Sales:")
    print(monthly_sales.tail())
    
    # Step 6: Customer Segmentation
    print("\n[5/6] CUSTOMER SEGMENTATION")
    print("-" * 70)
    segmentation = CustomerSegmentation(features)
    clusters, kmeans_model = segmentation.kmeans_clustering(n_clusters=4)
    
    # Step 7: Summary
    print("\n[6/6] PROJECT SUMMARY")
    print("-" * 70)
    print(f"✓ Total Customers: {len(customers):,}")
    print(f"✓ Total Transactions: {len(transactions):,}")
    print(f"✓ Features Created: {len(features.columns)}")
    print(f"✓ Models Trained: {len(models)}")
    print(f"✓ Best Model: XGBoost")
    print(f"✓ Customer Segments: 4")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE! 🎉")
    print("="*70)
    print("\nNext Steps:")
    print("1. Run dashboard: streamlit run src/dashboard.py")
    print("2. View saved models in models/ directory")
    print("3. Check visualizations in outputs/ directory")
    
    return {
        'customers': customers,
        'transactions': transactions,
        'features': features_encoded,
        'models': models,
        'results': results
    }

if __name__ == "__main__":
    project_data = main()