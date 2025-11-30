# ============================================================================
# STREAMLIT DASHBOARD FOR CUSTOMER CHURN PREDICTION
# File: dashboard.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_sample_data():
    """Load or generate sample data"""
    np.random.seed(42)
    
    # Generate customer data
    n_customers = 5000
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'tenure_months': np.random.randint(1, 72, n_customers),
        'age': np.random.randint(18, 80, n_customers),
        'monthly_charges': np.random.uniform(20, 200, n_customers),
        'total_charges': np.random.uniform(100, 10000, n_customers),
        'purchase_frequency': np.random.randint(0, 20, n_customers),
        'engagement_score': np.random.uniform(1, 10, n_customers),
        'churned': np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
    })
    
    # Generate monthly sales data
    months = pd.date_range('2023-01-01', '2024-12-31', freq='M')
    monthly_sales = pd.DataFrame({
        'month': months,
        'sales': np.random.uniform(45000, 85000, len(months)),
        'churn_rate': np.random.uniform(4, 18, len(months))
    })
    
    return customers, monthly_sales

def predict_churn_probability(tenure, monthly_charges, purchase_freq, engagement):
    """Simulate churn prediction"""
    # Simple weighted formula for demonstration
    base_prob = 50
    base_prob -= tenure * 0.8
    base_prob += (monthly_charges / 10) * 0.5
    base_prob -= purchase_freq * 3
    base_prob -= engagement * 2
    
    probability = max(5, min(95, base_prob))
    return probability

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 30:
        return "Low Risk", "🟢", "success"
    elif probability < 60:
        return "Medium Risk", "🟡", "warning"
    else:
        return "High Risk", "🔴", "error"

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    customers, monthly_sales = load_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://drive.google.com/file/d/1aIVPBYOGvjOa3nWXSniHjERVGlBtHJ6o/view?usp=drive_link", use_container_width=True)
        st.markdown("## 📊 Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Churn Prediction", "Sales Analytics", "Customer Segments", "Model Performance"]
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Customers", f"{len(customers):,}")
        st.metric("Active Churn Rate", f"{(customers['churned'].mean() * 100):.1f}%")
        st.metric("Avg Monthly Revenue", "$785K")
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "Overview":
        st.header("📊 Business Overview")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="👥 Total Customers",
                value=f"{len(customers):,}",
                delta="+12%"
            )
        
        with col2:
            churn_count = customers['churned'].sum()
            st.metric(
                label="⚠️ Churned Customers",
                value=f"{churn_count:,}",
                delta=f"-{(customers['churned'].mean() * 100):.1f}%",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="💰 Monthly Revenue",
                value="$785K",
                delta="+18%"
            )
        
        with col4:
            st.metric(
                label="🎯 Model Accuracy",
                value="91.2%",
                delta="XGBoost"
            )
        
        st.markdown("---")
        
        # Sales and Churn Trend
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Monthly Sales Trend")
            fig_sales = px.line(
                monthly_sales, 
                x='month', 
                y='sales',
                title='Sales Over Time',
                labels={'sales': 'Sales ($)', 'month': 'Month'}
            )
            fig_sales.update_traces(line_color='#667eea', line_width=3)
            fig_sales.update_layout(height=400)
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            st.subheader("📉 Churn Rate Trend")
            fig_churn = px.line(
                monthly_sales,
                x='month',
                y='churn_rate',
                title='Churn Rate Over Time',
                labels={'churn_rate': 'Churn Rate (%)', 'month': 'Month'}
            )
            fig_churn.update_traces(line_color='#f63366', line_width=3)
            fig_churn.update_layout(height=400)
            st.plotly_chart(fig_churn, use_container_width=True)
        
        # Customer Distribution
        st.subheader("👤 Customer Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tenure distribution
            fig_tenure = px.histogram(
                customers,
                x='tenure_months',
                nbins=30,
                title='Customer Tenure Distribution',
                labels={'tenure_months': 'Tenure (Months)'}
            )
            fig_tenure.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_tenure, use_container_width=True)
        
        with col2:
            # Monthly charges distribution
            fig_charges = px.histogram(
                customers,
                x='monthly_charges',
                nbins=30,
                title='Monthly Charges Distribution',
                labels={'monthly_charges': 'Monthly Charges ($)'}
            )
            fig_charges.update_traces(marker_color='#764ba2')
            st.plotly_chart(fig_charges, use_container_width=True)
    
    # ========================================================================
    # PAGE 2: CHURN PREDICTION
    # ========================================================================
    elif page == "Churn Prediction":
        st.header("🎯 Real-Time Churn Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Customer Features")
            
            # Input features
            tenure = st.slider("Tenure (months)", 1, 72, 24)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 200, 70)
            purchase_freq = st.slider("Purchase Frequency (per month)", 0, 20, 5)
            engagement = st.slider("Engagement Score (1-10)", 1.0, 10.0, 7.5, 0.5)
            
            st.markdown("---")
            
            # Model selection
            model_name = st.selectbox(
                "Select Prediction Model",
                ["XGBoost", "Random Forest", "LightGBM", "Neural Network", "Logistic Regression"]
            )
            
            # Predict button
            if st.button("🔮 Predict Churn", type="primary"):
                st.session_state['predicted'] = True
        
        with col2:
            st.subheader("📊 Prediction Results")
            
            if st.session_state.get('predicted', False):
                # Calculate prediction
                probability = predict_churn_probability(tenure, monthly_charges, purchase_freq, engagement)
                risk_level, emoji, status = get_risk_level(probability)
                
                # Display prediction gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability,
                    title={'text': f"{emoji} Churn Probability"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Risk assessment
                if status == "error":
                    st.error(f"{emoji} **{risk_level}** - Immediate action required!")
                elif status == "warning":
                    st.warning(f"{emoji} **{risk_level}** - Monitor closely")
                else:
                    st.success(f"{emoji} **{risk_level}** - Customer is healthy")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if probability > 60:
                    st.markdown("""
                    - 🎁 **Immediate**: Offer 20% retention discount
                    - 📞 **Contact**: Schedule personal call within 48 hours
                    - 🎯 **Target**: Address specific pain points
                    - 📧 **Follow-up**: Send personalized email campaign
                    """)
                elif probability > 30:
                    st.markdown("""
                    - 📊 **Survey**: Send satisfaction survey
                    - 🎓 **Education**: Offer product training session
                    - 🤝 **Engagement**: Increase touchpoint frequency
                    - 💬 **Feedback**: Request product feedback
                    """)
                else:
                    st.markdown("""
                    - ✅ **Maintain**: Continue current engagement level
                    - 📈 **Upsell**: Consider premium feature offerings
                    - 🌟 **Referral**: Request customer testimonial
                    - 🎁 **Reward**: Loyalty program enrollment
                    """)
            else:
                st.info("👈 Adjust customer features and click 'Predict Churn' to see results")
        
        # Batch prediction
        st.markdown("---")
        st.subheader("📤 Batch Prediction")
        st.markdown("Upload a CSV file with customer data for batch predictions")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview:", batch_data.head())
            
            if st.button("Run Batch Predictions"):
                # Simulate predictions
                batch_data['churn_probability'] = np.random.uniform(10, 90, len(batch_data))
                batch_data['risk_level'] = batch_data['churn_probability'].apply(
                    lambda x: 'High' if x > 60 else ('Medium' if x > 30 else 'Low')
                )
                
                st.success(f"✅ Processed {len(batch_data)} customers")
                st.dataframe(batch_data)
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # PAGE 3: SALES ANALYTICS
    # ========================================================================
    elif page == "Sales Analytics":
        st.header("💰 Sales Analytics")
        
        # Top products
        st.subheader("🏆 Top Performing Products")
        
        products_data = pd.DataFrame({
            'Product': ['Premium Plan', 'Standard Plan', 'Basic Plan', 'Add-ons', 'Enterprise'],
            'Revenue': [125000, 98000, 67000, 45000, 89000],
            'Customers': [850, 1400, 2200, 950, 320]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_products = px.bar(
                products_data,
                x='Product',
                y='Revenue',
                title='Revenue by Product',
                color='Revenue',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_products, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                products_data,
                values='Revenue',
                names='Product',
                title='Revenue Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Sales vs Churn correlation
        st.subheader("📊 Sales vs Churn Correlation")
        
        fig_corr = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_corr.add_trace(
            go.Bar(x=monthly_sales['month'], y=monthly_sales['sales'], name="Sales"),
            secondary_y=False,
        )
        
        fig_corr.add_trace(
            go.Scatter(x=monthly_sales['month'], y=monthly_sales['churn_rate'], 
                      name="Churn Rate", line=dict(color='red', width=3)),
            secondary_y=True,
        )
        
        fig_corr.update_xaxes(title_text="Month")
        fig_corr.update_yaxes(title_text="Sales ($)", secondary_y=False)
        fig_corr.update_yaxes(title_text="Churn Rate (%)", secondary_y=True)
        fig_corr.update_layout(height=500)
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **📈 Growth Trend**
            
            Revenue increased by 18% compared to last quarter, driven by Premium Plan adoption.
            """)
        
        with col2:
            st.success("""
            **🎯 Best Performer**
            
            Premium Plan generates highest revenue per customer at $147/month average.
            """)
        
        with col3:
            st.warning("""
            **🔄 Opportunity**
            
            Basic Plan has largest customer base - significant upsell potential.
            """)
    
    # ========================================================================
    # PAGE 4: CUSTOMER SEGMENTS
    # ========================================================================
    elif page == "Customer Segments":
        st.header("👥 Customer Segmentation")
        
        # Segment distribution
        segments = pd.DataFrame({
            'Segment': ['High Value', 'Medium Value', 'Low Value', 'At Risk'],
            'Customers': [2500, 4200, 1800, 1200],
            'Avg Revenue': [150, 80, 35, 60]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Segment Distribution")
            fig_segments = px.pie(
                segments,
                values='Customers',
                names='Segment',
                title='Customer Segments',
                color_discrete_sequence=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            st.subheader("💰 Revenue per Segment")
            fig_revenue = px.bar(
                segments,
                x='Segment',
                y='Avg Revenue',
                title='Average Revenue by Segment',
                color='Avg Revenue',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Segment details
        st.subheader("📋 Segment Details")
        st.dataframe(
            segments.style.background_gradient(cmap='YlGnBu', subset=['Customers', 'Avg Revenue']),
            use_container_width=True
        )
        
        # Scatter plot
        st.subheader("🎯 Customer Distribution (Tenure vs Charges)")
        fig_scatter = px.scatter(
            customers,
            x='tenure_months',
            y='monthly_charges',
            color='churned',
            size='total_charges',
            title='Customer Segmentation Visualization',
            labels={
                'tenure_months': 'Tenure (Months)',
                'monthly_charges': 'Monthly Charges ($)',
                'churned': 'Churned'
            },
            color_discrete_map={0: '#3b82f6', 1: '#ef4444'}
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # ========================================================================
    # PAGE 5: MODEL PERFORMANCE
    # ========================================================================
    elif page == "Model Performance":
        st.header("🤖 Model Performance Comparison")
        
        # Model metrics
        model_metrics = pd.DataFrame({
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                     'XGBoost', 'LightGBM', 'Neural Network'],
            'Accuracy': [78.5, 82.3, 89.7, 91.2, 90.8, 88.4],
            'Precision': [76.2, 80.1, 88.4, 90.5, 89.9, 87.1],
            'Recall': [74.8, 79.5, 87.9, 89.8, 89.2, 86.5],
            'F1 Score': [75.5, 79.8, 88.1, 90.1, 89.5, 86.8]
        })
        
        # Performance comparison
        st.subheader("📊 Model Metrics Comparison")
        
        fig_models = px.bar(
            model_metrics.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            title='Model Performance Across Metrics'
        )
        fig_models.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Best models
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **🥇 Best Overall**
            
            **XGBoost**
            - Accuracy: 91.2%
            - Best for production deployment
            """)
        
        with col2:
            st.info("""
            **⚡ Fastest**
            
            **Logistic Regression**
            - Quick training time
            - Good for rapid iterations
            """)
        
        with col3:
            st.warning("""
            **🎯 Most Balanced**
            
            **Random Forest**
            - Robust performance
            - Easy to interpret
            """)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Performance Metrics")
        st.dataframe(
            model_metrics.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1 Score']),
            use_container_width=True
        )
        
        # Feature importance (simulated)
        st.subheader("🎯 Feature Importance (XGBoost)")
        
        features_importance = pd.DataFrame({
            'Feature': ['Tenure', 'Monthly Charges', 'Purchase Frequency', 
                       'Engagement Score', 'Total Charges', 'Age'],
            'Importance': [0.28, 0.22, 0.18, 0.15, 0.12, 0.05]
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            features_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Scores',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()