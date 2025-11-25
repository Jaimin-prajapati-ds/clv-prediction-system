"""
CLV Prediction System - Streamlit Dashboard
==========================================

Interactive dashboard for Customer Lifetime Value analysis and prediction.

Author: Jaimin Prajapati
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="CLV Prediction System",
    page_icon="🎯",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load transaction and customer data."""
    try:
        transactions = pd.read_csv('data/transactions.csv')
        customers = pd.read_csv('data/customers.csv')
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        customers['join_date'] = pd.to_datetime(customers['join_date'])
        return transactions, customers
    except FileNotFoundError:
        # Generate sample data for demo purposes
        st.info("Loading demo data - generating sample customer transactions...")
        np.random.seed(42)
        n_customers = 500
        n_transactions = 5000
        
        # Generate customers
        customer_ids = [f'CUST_{i:04d}' for i in range(1, n_customers + 1)]
        join_dates = pd.date_range(start='2020-01-01', periods=n_customers, freq='D')
        customers = pd.DataFrame({
            'customer_id': customer_ids,
            'join_date': join_dates,
            'segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_customers),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers)
        })
        
        # Generate transactions
        transactions = pd.DataFrame({
            'transaction_id': [f'TXN_{i:06d}' for i in range(1, n_transactions + 1)],
            'customer_id': np.random.choice(customer_ids, n_transactions),
            'transaction_date': pd.date_range(start='2020-01-01', periods=n_transactions, freq='H'),
            'amount': np.random.exponential(50, n_transactions) + 10,
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Home'], n_transactions)
        })
        
        return transactions, customers

def calculate_rfm_scores(transactions):
    """Calculate RFM scores for customers."""
    reference_date = transactions['transaction_date'].max()
    
    rfm = transactions.groupby('customer_id').agg({
        'transaction_date': lambda x: (reference_date - x.max()).days,
        'customer_id': 'count',
        'amount': 'sum'
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm['avg_purchase'] = rfm['monetary'] / rfm['frequency']
    
    return rfm


def create_segment_distribution(rfm_data):
    """Create customer segment distribution chart."""
    # Simple segmentation
    def segment(row):
        if row['recency'] <= 30 and row['frequency'] >= 10:
            return 'Champions'
        elif row['recency'] <= 60 and row['frequency'] >= 5:
            return 'Loyal'
        elif row['recency'] <= 90:
            return 'Potential'
        elif row['recency'] <= 180:
            return 'At Risk'
        else:
            return 'Lost'
    
    rfm_data['segment'] = rfm_data.apply(segment, axis=1)
    
    segment_counts = rfm_data['segment'].value_counts()
    
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title='Customer Segment Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig, rfm_data


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">🎯 CLV Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    transactions, customers = load_data()
    
    if transactions is None or customers is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Customer Analytics", "RFM Segmentation", "CLV Prediction", "Category Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **About This Dashboard**
        
        Interactive tool for analyzing customer lifetime value 
        and segmenting customers based on purchase behavior.
        
        Built with Streamlit & Plotly
        """
    )
    
    # Calculate RFM
    rfm_data = calculate_rfm_scores(transactions)
    
    # PAGE 1: Overview
    if page == "Overview":
        st.header("📈 Business Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Customers",
                f"{customers.shape[0]:,}",
                delta="Active Users"
            )
        
        with col2:
            total_revenue = transactions['amount'].sum()
            st.metric(
                "Total Revenue",
                f"${total_revenue:,.2f}",
                delta=f"${total_revenue/len(customers):.2f} per customer"
            )
        
        with col3:
            avg_transaction = transactions['amount'].mean()
            st.metric(
                "Avg Transaction",
                f"${avg_transaction:.2f}",
                delta="Per order"
            )
        
        with col4:
            total_transactions = len(transactions)
            st.metric(
                "Total Transactions",
                f"{total_transactions:,}",
                delta=f"{total_transactions/len(customers):.1f} per customer"
            )
        
        st.markdown("---")
        
        # Revenue over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue Trend")
            daily_revenue = transactions.groupby(transactions['transaction_date'].dt.date)['amount'].sum().reset_index()
            fig_revenue = px.line(
                daily_revenue,
                x='transaction_date',
                y='amount',
                title='Daily Revenue',
                labels={'amount': 'Revenue ($)', 'transaction_date': 'Date'}
            )
            fig_revenue.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            st.subheader("📊 Transaction Volume")
            daily_trans = transactions.groupby(transactions['transaction_date'].dt.date).size().reset_index(name='count')
            fig_trans = px.bar(
                daily_trans,
                x='transaction_date',
                y='count',
                title='Daily Transactions',
                labels={'count': 'Number of Transactions', 'transaction_date': 'Date'}
            )
            fig_trans.update_traces(marker_color='#2ca02c')
            st.plotly_chart(fig_trans, use_container_width=True)
    
    # PAGE 2: Customer Analytics
    elif page == "Customer Analytics":
        st.header("👥 Customer Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Value Distribution")
            customer_value = transactions.groupby('customer_id')['amount'].sum()
            fig_dist = px.histogram(
                customer_value,
                nbins=50,
                title='Distribution of Customer Lifetime Value',
                labels={'value': 'Customer Lifetime Value ($)', 'count': 'Number of Customers'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Stats
            st.info(f"""
            **Key Statistics:**
            - Mean CLV: ${customer_value.mean():.2f}
            - Median CLV: ${customer_value.median():.2f}
            - Top 10% contribute: ${customer_value.quantile(0.9):.2f}+
            """)
        
        with col2:
            st.subheader("Purchase Frequency")
            customer_freq = transactions.groupby('customer_id').size()
            fig_freq = px.histogram(
                customer_freq,
                nbins=30,
                title='Distribution of Purchase Frequency',
                labels={'value': 'Number of Purchases', 'count': 'Number of Customers'}
            )
            st.plotly_chart(fig_freq, use_container_width=True)
            
            # Stats
            st.info(f"""
            **Frequency Statistics:**
            - Avg purchases per customer: {customer_freq.mean():.1f}
            - Single-purchase customers: {(customer_freq == 1).sum()} ({(customer_freq == 1).sum()/len(customer_freq)*100:.1f}%)
            - Repeat customers (5+): {(customer_freq >= 5).sum()}
            """)
    
    # PAGE 3: RFM Segmentation
    elif page == "RFM Segmentation":
        st.header("🗂️ RFM Customer Segmentation")
        
        fig_segments, rfm_with_segments = create_segment_distribution(rfm_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with col2:
            st.subheader("Segment Statistics")
            segment_stats = rfm_with_segments.groupby('segment').agg({
                'monetary': ['count', 'sum', 'mean'],
                'frequency': 'mean',
                'recency': 'mean'
            }).round(2)
            
            segment_stats.columns = ['Customers', 'Total Revenue', 'Avg CLV', 'Avg Frequency', 'Avg Recency']
            st.dataframe(segment_stats, use_container_width=True)
        
        # RFM Scatter
        st.subheader("RFM Analysis - 3D Visualization")
        fig_3d = px.scatter_3d(
            rfm_with_segments.reset_index(),
            x='recency',
            y='frequency',
            z='monetary',
            color='segment',
            title='Customer Segmentation in RFM Space',
            labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # PAGE 4: CLV Prediction
    elif page == "CLV Prediction":
        st.header("🔮 CLV Prediction")
        
        st.info("⚠️ Model training in progress. This feature will predict customer lifetime value using XGBoost.")
        
        st.subheader("Enter Customer Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
        
        with col2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=50, value=5)
        
        with col3:
            monetary = st.number_input("Monetary (total spending $)", min_value=0.0, value=500.0, step=10.0)
        
        if st.button("Predict CLV", type="primary"):
            # Simple prediction logic (placeholder)
            base_clv = monetary * (1 + frequency * 0.1) * (1 - recency/365)
            predicted_clv = max(base_clv, monetary)
            
            st.success(f"### Predicted Customer Lifetime Value: ${predicted_clv:.2f}")
            
            # Segment assignment
            if recency <= 30 and frequency >= 10:
                segment = "Champions"
                color = "green"
            elif recency <= 60 and frequency >= 5:
                segment = "Loyal"
                color = "blue"
            else:
                segment = "At Risk"
                color = "red"
            
            st.markdown(f"**Customer Segment:** :{color}[{segment}]")
    
    # PAGE 5: Category Analysis
    elif page == "Category Analysis":
        st.header("📦 Product Category Analysis")
        
        category_data = transactions.groupby('product_category').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        
        category_data.columns = ['Total Revenue', 'Avg Transaction', 'Transactions']
        category_data = category_data.sort_values('Total Revenue', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cat_revenue = px.bar(
                category_data.reset_index(),
                x='product_category',
                y='Total Revenue',
                title='Revenue by Product Category',
                color='Total Revenue',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cat_revenue, use_container_width=True)
        
        with col2:
            fig_cat_trans = px.bar(
                category_data.reset_index(),
                x='product_category',
                y='Transactions',
                title='Transaction Volume by Category',
                color='Transactions',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_cat_trans, use_container_width=True)
        
        st.subheader("Category Performance Table")
        st.dataframe(category_data, use_container_width=True)


if __name__ == '__main__':
    main()
