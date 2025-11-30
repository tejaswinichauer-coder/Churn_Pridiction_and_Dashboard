import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, customer_path='data/customers.csv', 
                  transaction_path='data/transactions.csv',
                  product_path='data/products.csv'):
        """Load datasets from CSV files"""
        try:
            customers = pd.read_csv(customer_path)
            transactions = pd.read_csv(transaction_path)
            
            try:
                products = pd.read_csv(product_path)
            except:
                products = None
                
            print(f"✓ Loaded {len(customers)} customers")
            print(f"✓ Loaded {len(transactions)} transactions")
            if products is not None:
                print(f"✓ Loaded {len(products)} products")
                
            return customers, transactions, products
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Generate customer data
        n_customers = 10000
        customers = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'tenure_months': np.random.randint(1, 72, n_customers),
            'age': np.random.randint(18, 80, n_customers),
            'monthly_charges': np.random.uniform(20, 200, n_customers),
            'total_charges': np.random.uniform(100, 10000, n_customers),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
            'online_security': np.random.choice(['Yes', 'No', 'No internet'], n_customers),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n_customers),
            'churned': np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
        })
        
        # Generate transaction data
        n_transactions = 50000
        transactions = pd.DataFrame({
            'transaction_id': range(1, n_transactions + 1),
            'customer_id': np.random.randint(1, n_customers + 1, n_transactions),
            'date': pd.date_range('2022-01-01', periods=n_transactions, freq='H'),
            'amount': np.random.uniform(10, 500, n_transactions),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_transactions)
        })
        
        # Generate product data
        products = pd.DataFrame({
            'product_id': range(1, 101),
            'product_name': [f'Product_{i}' for i in range(1, 101)],
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
            'price': np.random.uniform(10, 500, 100)
        })
        
        print("✓ Generated sample data")
        return customers, transactions, products
    
    def clean_data(self, df):
        """Handle missing values and outliers"""
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        print(f"✓ Cleaned data: {len(df)} records")
        return df
    
    def encode_categorical(self, df, columns):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def normalize_features(self, df, columns):
        """Normalize numerical features"""
        df_normalized = df.copy()
        df_normalized[columns] = self.scaler.fit_transform(df[columns])
        return df_normalized