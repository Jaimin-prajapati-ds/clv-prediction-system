"""
RFM Analysis Module for Customer Segmentation
============================================

This module performs Recency, Frequency, Monetary (RFM) analysis
to segment customers based on their purchase behavior.

Author: Jaimin Prajapati
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RFMAnalyzer:
    """
    Performs RFM analysis and customer segmentation.
    
    RFM scoring helps identify:
    - Champions: Best customers
    - Loyal Customers: Regular purchasers
    - At Risk: Previously good customers becoming inactive
    - Lost: Churned customers
    """
    
    def __init__(self, reference_date=None):
        """
        Initialize RFM analyzer.
        
        Args:
            reference_date: Date to calculate recency from (default: latest transaction date)
        """
        self.reference_date = reference_date
        self.rfm_df = None
        self.segments = None
        
    def calculate_rfm(self, transactions_df, customer_col='customer_id', 
                     date_col='transaction_date', amount_col='amount'):
        """
        Calculate RFM metrics for each customer.
        
        Args:
            transactions_df: DataFrame with transaction data
            customer_col: Column name for customer ID
            date_col: Column name for transaction date
            amount_col: Column name for transaction amount
            
        Returns:
            DataFrame with RFM scores for each customer
        """
        print("📊 Calculating RFM metrics...")
        
        # Convert date column to datetime if needed
        transactions_df[date_col] = pd.to_datetime(transactions_df[date_col])
        
        # Set reference date if not provided
        if self.reference_date is None:
            self.reference_date = transactions_df[date_col].max()
        
        # Calculate RFM metrics
        rfm = transactions_df.groupby(customer_col).agg({
            date_col: lambda x: (self.reference_date - x.max()).days,  # Recency
            customer_col: 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        })
        
        # Rename columns
        rfm.columns = ['recency', 'frequency', 'monetary']
        
        # Calculate RFM scores (1-5 scale, 5 being best)
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5])
        
        # Convert scores to integers
        rfm['r_score'] = rfm['r_score'].astype(int)
        rfm['f_score'] = rfm['f_score'].astype(int)
        rfm['m_score'] = rfm['m_score'].astype(int)
        
        # Calculate combined RFM score
        rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
        
        # Create concatenated score for segmentation
        rfm['rfm_segment'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        self.rfm_df = rfm.reset_index()
        
        print(f"✅ RFM calculated for {len(rfm)} customers")
        return self.rfm_df
    
    def segment_customers(self):
        """
        Segment customers based on RFM scores.
        
        Returns:
            DataFrame with customer segments
        """
        if self.rfm_df is None:
            raise ValueError("Calculate RFM first using calculate_rfm()")
        
        print("🎯 Segmenting customers...")
        
        df = self.rfm_df.copy()
        
        # Define segmentation logic
        def assign_segment(row):
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            score = row['rfm_score']
            
            # Champions: High R, F, M
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            
            # Loyal Customers: High F and M, moderate R
            elif f >= 4 and m >= 4:
                return 'Loyal Customers'
            
            # Potential Loyalists: Recent customers with average frequency
            elif r >= 4 and f >= 2 and f <= 3:
                return 'Potential Loyalists'
            
            # Recent Customers: Very recent but low frequency/monetary
            elif r >= 4 and f == 1:
                return 'Recent Customers'
            
            # Promising: Recent with moderate spending
            elif r >= 3 and m >= 3:
                return 'Promising'
            
            # Need Attention: Above average recency, frequency & monetary
            elif r == 3 and f >= 3 and m >= 3:
                return 'Need Attention'
            
            # At Risk: Good customers who haven't purchased recently
            elif r <= 2 and f >= 4 and m >= 4:
                return 'At Risk'
            
            # Can't Lose Them: High spenders who haven't returned
            elif r <= 2 and f >= 3 and m >= 4:
                return "Can't Lose Them"
            
            # Hibernating: Low recency, once good customers
            elif r <= 2 and f >= 2 and m >= 2:
                return 'Hibernating'
            
            # Lost: Lowest recency, frequency & monetary
            else:
                return 'Lost'
        
        df['segment'] = df.apply(assign_segment, axis=1)
        
        self.segments = df
        
        # Print segment distribution
        segment_counts = df['segment'].value_counts()
        print("\n📈 Customer Segment Distribution:")
        for segment, count in segment_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {segment}: {count} customers ({pct:.1f}%)")
        
        return self.segments
    
    def get_segment_statistics(self):
        """
        Get statistics for each customer segment.
        
        Returns:
            DataFrame with segment statistics
        """
        if self.segments is None:
            raise ValueError("Segment customers first using segment_customers()")
        
        stats = self.segments.groupby('segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['mean', 'sum']
        }).round(2)
        
        stats.columns = ['customer_count', 'avg_recency', 'avg_frequency', 'avg_monetary', 'total_revenue']
        stats = stats.reset_index()
        
        # Calculate revenue percentage
        total_revenue = stats['total_revenue'].sum()
        stats['revenue_pct'] = ((stats['total_revenue'] / total_revenue) * 100).round(1)
        
        # Sort by total revenue
        stats = stats.sort_values('total_revenue', ascending=False)
        
        return stats
    
    def get_actionable_insights(self):
        """
        Generate actionable business insights for each segment.
        
        Returns:
            Dictionary with segment recommendations
        """
        insights = {
            'Champions': {
                'description': 'Your best customers with high R, F, and M scores',
                'action': 'Reward them. Offer exclusive benefits and early access to new products.',
                'priority': 'High'
            },
            'Loyal Customers': {
                'description': 'Regular purchasers with good spending habits',
                'action': 'Upsell higher value products. Ask for reviews and referrals.',
                'priority': 'High'
            },
            'Potential Loyalists': {
                'description': 'Recent customers with moderate engagement',
                'action': 'Nurture with personalized recommendations and loyalty programs.',
                'priority': 'Medium'
            },
            'Recent Customers': {
                'description': 'New customers who just made their first purchase',
                'action': 'Provide excellent onboarding experience. Send welcome offers.',
                'priority': 'Medium'
            },
            'At Risk': {
                'description': 'Good customers who haven\'t purchased recently',
                'action': 'Send win-back campaigns. Offer special discounts.',
                'priority': 'High'
            },
            "Can't Lose Them": {
                'description': 'High-value customers slipping away',
                'action': 'Aggressive retention campaign. Personal outreach required.',
                'priority': 'Critical'
            },
            'Hibernating': {
                'description': 'Customers who have significantly reduced activity',
                'action': 'Re-engagement campaigns with compelling offers.',
                'priority': 'Medium'
            },
            'Lost': {
                'description': 'Customers who have likely churned',
                'action': 'Revive with special offers or accept loss and focus on acquisition.',
                'priority': 'Low'
            }
        }
        
        return insights
    
    def export_segments(self, filepath='data/customer_segments.csv'):
        """
        Export segmented customer data to CSV.
        
        Args:
            filepath: Path to save the CSV file
        """
        if self.segments is None:
            raise ValueError("Segment customers first using segment_customers()")
        
        self.segments.to_csv(filepath, index=False)
        print(f"✅ Segments exported to {filepath}")


def main():
    """
    Example usage of RFM Analyzer.
    """
    print("="*60)
    print(" RFM Analysis - Customer Segmentation")
    print("="*60)
    print()
    
    # Load transaction data
    print("📂 Loading transaction data...")
    transactions = pd.read_csv('data/transactions.csv')
    
    # Initialize analyzer
    analyzer = RFMAnalyzer()
    
    # Calculate RFM
    rfm_df = analyzer.calculate_rfm(transactions)
    
    # Segment customers
    segments_df = analyzer.segment_customers()
    
    # Get statistics
    print("\n" + "="*60)
    stats = analyzer.get_segment_statistics()
    print("\n📊 Segment Statistics:")
    print(stats.to_string(index=False))
    
    # Export results
    analyzer.export_segments()
    
    print("\n✨ RFM Analysis Complete!")


if __name__ == '__main__':
    main()