import matplotlib.pyplot as plt
import pandas as pd

class SalesAnalyzer:
    """Analyze sales trends and patterns"""
    
    def __init__(self, transactions):
        self.transactions = transactions
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        
    def monthly_sales_trend(self):
        """Calculate monthly sales"""
        self.transactions['month'] = self.transactions['date'].dt.to_period('M')
        monthly = self.transactions.groupby('month')['amount'].agg(['sum', 'count']).reset_index()
        monthly.columns = ['month', 'total_sales', 'transaction_count']
        monthly['month'] = monthly['month'].astype(str)
        return monthly
    
    def quarterly_sales(self):
        """Calculate quarterly sales"""
        self.transactions['quarter'] = self.transactions['date'].dt.to_period('Q')
        quarterly = self.transactions.groupby('quarter')['amount'].sum().reset_index()
        quarterly.columns = ['quarter', 'total_sales']
        quarterly['quarter'] = quarterly['quarter'].astype(str)
        return quarterly
    
    def top_products(self, n=10):
        """Identify top performing products"""
        if 'product_category' in self.transactions.columns:
            top = self.transactions.groupby('product_category').agg({
                'amount': ['sum', 'count', 'mean']
            }).reset_index()
            top.columns = ['product', 'total_revenue', 'num_transactions', 'avg_transaction']
            top = top.sort_values('total_revenue', ascending=False).head(n)
            return top
        return None
    
    def churn_revenue_correlation(self, customers):
        """Analyze correlation between churn and revenue"""
        # Merge transactions with customer churn status
        merged = pd.merge(
            self.transactions,
            customers[['customer_id', 'churned']],
            on='customer_id',
            how='left'
        )
        
        churn_stats = merged.groupby('churned')['amount'].agg(['sum', 'count', 'mean']).reset_index()
        churn_stats.columns = ['churned', 'total_revenue', 'num_transactions', 'avg_transaction']
        
        return churn_stats
    
    def plot_trends(self, save_path='outputs/'):
        """Create visualization plots"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Monthly sales trend
        monthly = self.monthly_sales_trend()
        
        plt.figure(figsize=(12, 6))
        plt.plot(monthly['month'], monthly['total_sales'], marker='o', linewidth=2)
        plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Total Sales ($)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}monthly_sales.png', dpi=300)
        plt.close()
        
        print(f"✓ Saved sales visualizations to {save_path}")