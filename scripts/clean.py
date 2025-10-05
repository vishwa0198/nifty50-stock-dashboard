import pandas as pd
import numpy as np
from datetime import datetime
import os

class NiftyDataCleaner:
    def __init__(self, data_dir=r"extracted"):
        self.data_dir = data_dir
    
    def load_data(self, file_path=None):
        if file_path:
            df = pd.read_csv(file_path)
        else:
            combined_path = r"nifty50_combined.csv"
            if os.path.exists(combined_path):
                df = pd.read_csv(combined_path)
            else:
                print("Combined dataset not found")
                return None
        
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} records for {df['ticker'].nunique()} stocks")
        return df
    
    def clean_data(self, df):
        df = df.drop_duplicates(subset=['ticker', 'date'])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
        
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        df = df[df[['open', 'high', 'low', 'close']].min(axis=1) > 0]
        df = df[df['volume'] > 0]
        
        for col in numeric_cols:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        
        print(f"Cleaned data: {len(df)} records")
        return df
    
    def add_technical_indicators(self, df):
        df = df.sort_values(['ticker', 'date'])
        
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            stock_data = df[mask].copy()
            
            stock_data['daily_return'] = stock_data['close'].pct_change()
            stock_data['cumulative_return'] = (1 + stock_data['daily_return']).cumprod() - 1
            
            stock_data['ma_5'] = stock_data['close'].rolling(5).mean()
            stock_data['ma_20'] = stock_data['close'].rolling(20).mean()
            stock_data['ma_50'] = stock_data['close'].rolling(50).mean()
            
            stock_data['volatility'] = stock_data['daily_return'].rolling(20).std() * np.sqrt(252)
            
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            stock_data['rsi'] = 100 - (100 / (1 + rs))
            
            stock_data['bb_middle'] = stock_data['close'].rolling(20).mean()
            bb_std = stock_data['close'].rolling(20).std()
            stock_data['bb_upper'] = stock_data['bb_middle'] + (bb_std * 2)
            stock_data['bb_lower'] = stock_data['bb_middle'] - (bb_std * 2)
            
            df.loc[mask, stock_data.columns] = stock_data
        
        print("Added technical indicators")
        return df
    
    def calculate_yearly_metrics(self, df):
        metrics = []
        
        for ticker in df['ticker'].unique():
            stock_data = df[df['ticker'] == ticker].sort_values('date')
            
            if len(stock_data) < 2:
                continue
            
            start_price = stock_data['close'].iloc[0]
            end_price = stock_data['close'].iloc[-1]
            yearly_return = ((end_price - start_price) / start_price) * 100
            
            avg_price = stock_data['close'].mean()
            avg_volume = stock_data['volume'].mean()
            avg_volatility = stock_data['volatility'].mean()
            
            cumulative_returns = (1 + stock_data['daily_return']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            risk_free_rate = 0.05
            excess_return = stock_data['daily_return'].mean() * 252 - risk_free_rate
            sharpe_ratio = excess_return / (stock_data['daily_return'].std() * np.sqrt(252)) if stock_data['daily_return'].std() > 0 else 0
            
            metrics.append({
                'ticker': ticker,
                'sector': stock_data['sector'].iloc[0],
                'yearly_return': yearly_return,
                'avg_price': avg_price,
                'avg_volume': avg_volume,
                'avg_volatility': avg_volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trading_days': len(stock_data)
            })
        
        yearly_metrics = pd.DataFrame(metrics)
        print(f"Calculated yearly metrics for {len(yearly_metrics)} stocks")
        return yearly_metrics
    
    def save_cleaned_data(self, df):
        cleaned_path = r"nifty50_cleaned.csv"
        df.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned data to {cleaned_path}")
    
    def run_cleaning_pipeline(self, input_file=None):
        print("Starting data cleaning pipeline...")
        
        df = self.load_data(input_file)
        if df is None:
            return None, None
        
        cleaned_df = self.clean_data(df)
        cleaned_df = self.add_technical_indicators(cleaned_df)
        yearly_metrics = self.calculate_yearly_metrics(cleaned_df)
        
        self.save_cleaned_data(cleaned_df)
        
        yearly_path = os.path.join(self.data_dir, "nifty50_yearly_metrics.csv")
        yearly_metrics.to_csv(yearly_path, index=False)
        print(f"Saved yearly metrics to {yearly_path}")
        
        print("Data cleaning pipeline completed!")
        return cleaned_df, yearly_metrics

def main():
    cleaner = NiftyDataCleaner()
    return cleaner.run_cleaning_pipeline()

if __name__ == "__main__":
    main()