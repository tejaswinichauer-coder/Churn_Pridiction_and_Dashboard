import pandas as pd
import numpy as np

class FeatureEngineer:
    """Creates advanced features for ML models"""
    
    def create_customer_features(self, customers, transactions):
        """Engineer features from customer and transaction data"""
        
        # Merge datasets
        merged = pd.merge(transactions, customers, on='customer_id', how='left')
        
        # Purchase frequency
        purchase_freq = transactions.groupby('customer_id').size().reset_index(name='purchase_frequency')
        
        # Average transaction amount
        avg_amount = transactions.groupby('customer_id')['amount'].mean().reset_index(name='avg_transaction_amount')
        
        # Total spent
        total_spent = transactions.groupby('customer_id')['amount'].sum().reset_index(name='total_spent')
        
        # Days since last purchase
        transactions['date'] = pd.to_datetime(transactions['date'])
        last_purchase = transactions.groupby('customer_id')['date'].max().reset_index(name='last_purchase_date')
        last_purchase['days_since_last_purchase'] = (pd.Timestamp.now() - last_purchase['last_purchase_date']).dt.days
        
        # Engagement score (composite metric)
        engagement = purchase_freq.copy()
        engagement = engagement.merge(avg_amount, on='customer_id')
        engagement['engagement_score'] = (
            engagement['purchase_frequency'] * 0.5 + 
            (engagement['avg_transaction_amount'] / 100) * 0.5
        )
        
        # Merge all features
        features = customers.copy()
        features = features.merge(purchase_freq, on='customer_id', how='left')
        features = features.merge(avg_amount, on='customer_id', how='left')
        features = features.merge(total_spent, on='customer_id', how='left')
        features = features.merge(last_purchase[['customer_id', 'days_since_last_purchase']], on='customer_id', how='left')
        features = features.merge(engagement[['customer_id', 'engagement_score']], on='customer_id', how='left')
        
        # Fill NaN values
        features = features.fillna(0)
        
        print(f"✓ Created {len(features.columns)} features")
        return features
    
    def create_time_features(self, transactions):
        """Create time-based features"""
        transactions['date'] = pd.to_datetime(transactions['date'])
        transactions['month'] = transactions['date'].dt.month
        transactions['day_of_week'] = transactions['date'].dt.dayofweek
        transactions['quarter'] = transactions['date'].dt.quarter
        transactions['year'] = transactions['date'].dt.year
        
        return transactions