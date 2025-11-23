# generate_data.py
# This script creates fake e-commerce data for my CLV prediction project
# Made for testing the RFM analysis and ML models

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_customer_data(n_customers=5000):
    Generate basic customer demographic data 
    
    print(f"📊 Generating {n_customers} customers...")
    
    # Customer segments with different behaviors
        # trying different customer types for the simulation
    segments = ['High-Value', 'Medium-Value', 'Low-Value', 'Churned']
    segment_weights = [0.15, 0.35, 0.40, 0.10]  # Distribution of customer types
    
    customers = {
        'customer_id': [f'CUST_{str(i).zfill(5)}' for i in range(1, n_customers + 1)],
        'join_date': pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_customers),
        'segment': np.random.choice(segments, n_customers, p=segment_weights),
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'Germany'], n_customers),
        'age': np.random.randint(18, 70, n_customers)
    }
    
    return pd.DataFrame(customers)

def generate_transactions(customers_df, avg_transactions_per_customer=8):
    Generate realistic transaction data based on customer segments.
    
    """
    print(f"💳 Generating transactions...")
    
    transactions = []
    transaction_id = 1
    
    # Product categories with price ranges
    product_categories = {
        'Electronics': (50, 500),
        'Clothing': (20, 150),
        'Home & Garden': (15, 200),
        'Books': (10, 40),
        'Sports': (25, 300),
        'Beauty': (15, 100),
        'Toys': (10, 80)
    }
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['customer_id']
        segment = customer['segment']
        join_date = customer['join_date']
        
        # Different transaction patterns based on segment
        if segment == 'High-Value':
            n_transactions = np.random.randint(15, 30)
            purchase_frequency_days = 15  # Shops more frequently
            value_multiplier = 2.0
        elif segment == 'Medium-Value':
            n_transactions = np.random.randint(5, 15)
            purchase_frequency_days = 30
            value_multiplier = 1.0
        elif segment == 'Low-Value':
            n_transactions = np.random.randint(1, 5)
            purchase_frequency_days = 60
            value_multiplier = 0.6
        else:  # Churned
            n_transactions = np.random.randint(1, 3)
            purchase_frequency_days = 90
            value_multiplier = 0.5
        
        # Generate transactions for this customer
        current_date = join_date
        max_date = datetime(2024, 11, 1)  # Cutoff date
        
        for _ in range(n_transactions):
            if current_date > max_date:
                break
            
            # Select random product category
            category = random.choice(list(product_categories.keys()))
            price_range = product_categories[category]
            
            # Generate transaction amount with some randomness
            base_amount = np.random.uniform(price_range[0], price_range[1])
            transaction_amount = round(base_amount * value_multiplier, 2)
            
            # Quantity (usually 1-3 items per transaction)
            quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
            
            transactions.append({
                'transaction_id': f'TXN_{str(transaction_id).zfill(6)}',
                'customer_id': customer_id,
                'transaction_date': current_date,
                'product_category': category,
                'quantity': quantity,
                'amount': round(transaction_amount * quantity, 2),
                'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Gift Card'])
            })
            
            transaction_id += 1
            
            # Add time gap between transactions with some randomness
            days_gap = np.random.exponential(purchase_frequency_days)
            current_date += timedelta(days=int(days_gap))
    
    df = pd.DataFrame(transactions)
    print(f"✅ Generated {len(df)} transactions")
    return df

def add_data_quality_issues(df, missing_rate=0.02):
    """
    
    print(f"🔧 Adding realistic data quality issues...")
    
    df_copy = df.copy()
    
    # Introduce some missing values in non-critical columns
    n_missing = int(len(df_copy) * missing_rate)
    missing_idx = np.random.choice(df_copy.index, n_missing, replace=False)
    df_copy.loc[missing_idx, 'payment_method'] = np.nan
    
    # Add a few outliers (returns/refunds as negative amounts)
    n_outliers = int(len(df_copy) * 0.01)
    outlier_idx = np.random.choice(df_copy.index, n_outliers, replace=False)
    df_copy.loc[outlier_idx, 'amount'] = df_copy.loc[outlier_idx, 'amount'] * -1
    
    return df_copy

def save_datasets(customers_df, transactions_df, output_dir='data'):
    """
    
    print(f"💾 Saving datasets to {output_dir}/...")
    
    customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
    transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
    
    print("✅ Data generation complete!")
    print(f"   - Customers: {len(customers_df)} records")
    print(f"   - Transactions: {len(transactions_df)} records")
    print(f"   - Date range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
    print(f"   - Total revenue: ${transactions_df['amount'].sum():,.2f}")

def main():
    """
    Main execution function to generate complete dataset.
    """
    print("="*60)
    print(" CLV Prediction System - Data Generation")
    print("="*60)
    print()
    
    # Generate customer data
    customers_df = generate_customer_data(n_customers=5000)
    
    # Generate transaction data
    transactions_df = generate_transactions(customers_df, avg_transactions_per_customer=8)
    
    # Add realistic data quality issues
    transactions_df = add_data_quality_issues(transactions_df, missing_rate=0.02)
    
    # Save datasets
    save_datasets(customers_df, transactions_df)
    
    print("\n🎉 Dataset ready for analysis!")
    print("\nNext steps:")
    print("  1. Run EDA notebook: notebooks/01_eda.ipynb")
    print("  2. Perform RFM analysis")
    print("  3. Train CLV prediction models")

if __name__ == '__main__':
    main()