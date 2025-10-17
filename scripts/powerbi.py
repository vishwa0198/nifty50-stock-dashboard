import pandas as pd
import numpy as np
import os
from datetime import datetime

class PowerBIDataExporter:
    def __init__(self):
        self.output_dir = r"powerbi_data"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_for_powerbi(self):
        # Load cleaned data
        df = pd.read_csv(r"nifty50_cleaned.csv")
        yearly_metrics = pd.read_csv(r"nifty50_yearly_metrics.csv")
        
        # Export tables for Power BI
        df.to_csv(f"{self.output_dir}/stock_data.csv", index=False)
        yearly_metrics.to_csv(f"{self.output_dir}/yearly_metrics.csv", index=False)
        
        # Create additional tables for Power BI relationships
        sectors = df[['ticker', 'sector']].drop_duplicates()
        sectors.to_csv(f"{self.output_dir}/sectors.csv", index=False)

        # Create Date Dimension
        df['date'] = pd.to_datetime(df['date'])
        start_date = df['date'].min().date()
        end_date = df['date'].max().date()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_dim = pd.DataFrame({'date': date_range})
        date_dim['year'] = date_dim['date'].dt.year
        date_dim['quarter'] = date_dim['date'].dt.quarter
        date_dim['month'] = date_dim['date'].dt.month
        date_dim['month_name'] = date_dim['date'].dt.strftime('%b')
        date_dim['year_month'] = date_dim['date'].dt.strftime('%Y-%m')
        date_dim.to_csv(f"{self.output_dir}/date_dim.csv", index=False)

        # Monthly Returns per Ticker
        df_sorted = df.sort_values(['ticker', 'date'])
        df_sorted['prev_close'] = df_sorted.groupby('ticker')['close'].shift(1)
        df_sorted['daily_return'] = (df_sorted['close'] - df_sorted['prev_close']) / df_sorted['prev_close']
        df_sorted['year_month'] = df_sorted['date'].dt.to_period('M').astype(str)
        monthly = (
            df_sorted.groupby(['ticker', 'year_month'])
            .agg(first_close=('close', 'first'), last_close=('close', 'last'))
            .reset_index()
        )
        monthly['monthly_return'] = ((monthly['last_close'] - monthly['first_close']) / monthly['first_close']) * 100
        monthly.to_csv(f"{self.output_dir}/monthly_returns.csv", index=False)

        # Correlation Matrix (based on daily returns)
        price_pivot = df_sorted.pivot_table(index='date', columns='ticker', values='close', aggfunc='first')
        returns = price_pivot.pct_change().dropna(how='all')
        corr = returns.corr()
        corr.to_csv(f"{self.output_dir}/correlation_matrix.csv")

        # Confirm outputs
        outputs = [
            "stock_data.csv",
            "yearly_metrics.csv",
            "sectors.csv",
            "date_dim.csv",
            "monthly_returns.csv",
            "correlation_matrix.csv",
        ]
        print("Data exported for Power BI import:")
        for name in outputs:
            path = os.path.join(self.output_dir, name)
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            print(f" - {path} {'(OK)' if exists else '(MISSING)'} size={size} bytes")
        return [os.path.join(self.output_dir, n) for n in outputs]

if __name__ == "__main__":
    exporter = PowerBIDataExporter()
    exporter.export_for_powerbi()
