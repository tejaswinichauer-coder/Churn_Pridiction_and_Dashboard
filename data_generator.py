import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SampleDataGenerator:
    """Generate realistic sample data for testing and demonstration"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def generate_customers(self, n=10000):
        """Generate customer dataset"""
        
        # Contract types and their churn probabilities
        contract_types = ['Month-to-month', 'One year', 'Two year']
        contract_weights = [0.5, 0.3, 0.2]
        
        customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(1, n + 1)],
            'age': np.random.randint(18, 80, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'tenure_months': np.random.randint(1, 72, n),
            'contract_type': np.random.choice(contract_types, n, p=contract_weights),
            'payment_method': np.random.choice(
                ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                n, 
                p=[0.4, 0.2, 0.2, 0.2]
            ),
            'monthly_charges': np.random.uniform(20, 200, n),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.3, 0.5, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet'], n, p=[0.3, 0.5, 0.2]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n, p=[0.35, 0.45, 0.2]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet'], n, p=[0.4, 0.4, 0.2]),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet'], n, p=[0.4, 0.4, 0.2]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4]),
        })
        
        # Calculate total charges based on tenure and monthly charges
        customers['total_charges'] = customers['tenure_months'] * customers['monthly_charges']
        customers['total_charges'] += np.random.uniform(-100, 500, n)  # Add some variance
        customers['total_charges'] = customers['total_charges'].clip(lower=0)
        
        # Generate churn based on realistic factors
        churn_prob = np.zeros(n)
        
        # Higher churn for month-to-month contracts
        churn_prob += (customers['contract_type'] == 'Month-to-month').astype(int) * 0.3
        
        # Higher churn for electronic check payments
        churn_prob += (customers['payment_method'] == 'Electronic check').astype(int) * 0.15
        
        # Lower churn for longer tenure
        churn_prob -= (customers['tenure_months'] / 72) * 0.25
        
        # Higher churn for higher monthly charges
        churn_prob += (customers['monthly_charges'] > 100).astype(int) * 0.1
        
        # Lower churn with tech support
        churn_prob -= (customers['tech_support'] == 'Yes').astype(int) * 0.15
        
        # Add random noise
        churn_prob += np.random.uniform(-0.1, 0.1, n)
        churn_prob = np.clip(churn_prob, 0, 1)
        
        # Generate binary churn outcome
        customers['churned'] = (np.random.random(n) < churn_prob).astype(int)
        
        print(f"✓ Generated {n} customers")
        print(f"  Churn rate: {customers['churned'].mean():.2%}")
        
        return customers
    
    def generate_transactions(self, customers, avg_transactions_per_customer=10):
        """Generate transaction dataset"""
        
        transactions = []
        transaction_id = 1
        
        for _, customer in customers.iterrows():
            # Number of transactions depends on tenure and churn status
            if customer['churned']:
                n_trans = max(1, int(np.random.poisson(avg_transactions_per_customer * 0.5)))
            else:
                n_trans = int(np.random.poisson(avg_transactions_per_customer))
            
            # Generate transaction dates within tenure period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=customer['tenure_months'] * 30)
            
            for _ in range(n_trans):
                transaction_date = start_date + timedelta(
                    days=np.random.randint(0, (end_date - start_date).days + 1)
                )
                
                transactions.append({
                    'transaction_id': f'TXN_{transaction_id:08d}',
                    'customer_id': customer['customer_id'],
                    'date': transaction_date,
                    'amount': abs(np.random.normal(customer['monthly_charges'] * 0.3, 20)),
                    'product_category': np.random.choice([
                        'Service Fee', 'Equipment', 'Add-on', 'Upgrade', 'Installation'
                    ], p=[0.6, 0.15, 0.15, 0.05, 0.05]),
                    'transaction_type': np.random.choice(['Purchase', 'Refund'], p=[0.95, 0.05])
                })
                transaction_id += 1
        
        transactions_df = pd.DataFrame(transactions)
        
        # Adjust refund amounts
        refund_mask = transactions_df['transaction_type'] == 'Refund'
        transactions_df.loc[refund_mask, 'amount'] *= -1
        
        print(f"✓ Generated {len(transactions_df)} transactions")
        print(f"  Avg per customer: {len(transactions_df) / len(customers):.1f}")
        
        return transactions_df
    
    def generate_products(self, n=100):
        """Generate product catalog"""
        
        categories = ['Internet Service', 'TV Package', 'Phone Service', 'Equipment', 'Add-ons']
        
        products = pd.DataFrame({
            'product_id': [f'PROD_{i:04d}' for i in range(1, n + 1)],
            'product_name': [f'Product {i}' for i in range(1, n + 1)],
            'category': np.random.choice(categories, n),
            'base_price': np.random.uniform(10, 150, n),
            'popularity_score': np.random.uniform(1, 10, n)
        })
        
        print(f"✓ Generated {n} products")
        
        return products
    
    def save_datasets(self, customers, transactions, products, path='data/'):
        """Save all datasets to CSV files"""
        import os
        os.makedirs(path, exist_ok=True)
        
        customers.to_csv(f'{path}customers.csv', index=False)
        transactions.to_csv(f'{path}transactions.csv', index=False)
        products.to_csv(f'{path}products.csv', index=False)
        
        print(f"\n✓ Saved all datasets to {path}")
        print(f"  - customers.csv ({len(customers)} rows)")
        print(f"  - transactions.csv ({len(transactions)} rows)")
        print(f"  - products.csv ({len(products)} rows)")
