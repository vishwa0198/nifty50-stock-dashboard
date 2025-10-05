import os
import yaml
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime

class NiftyDataExtractor:
    def __init__(self, raw_dir=r"C:\nifty50-stock-dashboard\data", output_dir="extracted"):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.data_by_symbol = defaultdict(list)
        self.sector_mapping = {
            'RELIANCE': 'Energy', 'TCS': 'Information Technology', 'HDFCBANK': 'Financial Services',
            'INFY': 'Information Technology', 'HINDUNILVR': 'Consumer Goods', 'ITC': 'Consumer Goods',
            'KOTAKBANK': 'Financial Services', 'HDFCLIFE': 'Financial Services', 'SBILIFE': 'Financial Services',
            'BHARTIARTL': 'Telecommunications', 'LT': 'Construction', 'ASIANPAINT': 'Consumer Goods',
            'AXISBANK': 'Financial Services', 'MARUTI': 'Automobile', 'TITAN': 'Consumer Goods',
            'NESTLEIND': 'Consumer Goods', 'ULTRACEMCO': 'Materials', 'WIPRO': 'Information Technology',
            'SUNPHARMA': 'Healthcare', 'TATAMOTORS': 'Automobile', 'POWERGRID': 'Utilities',
            'NTPC': 'Utilities', 'ONGC': 'Energy', 'COALINDIA': 'Energy', 'TECHM': 'Information Technology',
            'HCLTECH': 'Information Technology', 'BAJFINANCE': 'Financial Services', 'BAJAJFINSV': 'Financial Services',
            'BAJAJ-AUTO': 'Automobile', 'M&M': 'Automobile', 'TATASTEEL': 'Materials', 'JSWSTEEL': 'Materials',
            'HINDALCO': 'Materials', 'GRASIM': 'Materials', 'EICHERMOT': 'Automobile', 'HEROMOTOCO': 'Automobile',
            'DRREDDY': 'Healthcare', 'CIPLA': 'Healthcare', 'APOLLOHOSP': 'Healthcare', 'BRITANNIA': 'Consumer Goods',
            'INDUSINDBK': 'Financial Services', 'ICICIBANK': 'Financial Services', 'SBIN': 'Financial Services',
            'ADANIENT': 'Materials', 'ADANIPORTS': 'Services', 'BEL': 'Services', 'BPCL': 'Energy',
            'SHRIRAMFIN': 'Financial Services', 'TRENT': 'Consumer Services'
        }
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            if not isinstance(content, list):
                return
            
            for row in content:
                symbol = row.get('Ticker') or row.get('symbol') or row.get('ticker')
                if not symbol:
                    continue
                
                date_str = row.get('date', '')
                if not date_str:
                    continue
                    
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        continue
                
                try:
                    open_price = float(row.get('open', 0))
                    high_price = float(row.get('high', 0))
                    low_price = float(row.get('low', 0))
                    close_price = float(row.get('close', 0))
                    volume = int(row.get('volume', 0))
                    
                    if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
                        continue
                        
                except (ValueError, TypeError):
                    continue
                
                sector = self.sector_mapping.get(symbol, 'Unknown')
                
                self.data_by_symbol[symbol].append({
                    'ticker': symbol, 'date': date_obj, 'open': open_price,
                    'high': high_price, 'low': low_price, 'close': close_price,
                    'volume': volume, 'sector': sector
                })
                
        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
    
    def extract_data(self):
        yaml_count = 0
        for root, _, files in os.walk(self.raw_dir):
            for filename in files:
                if filename.endswith(('.yml', '.yaml')):
                    self.parse_file(os.path.join(root, filename))
                    yaml_count += 1
        
        print(f"Processed {yaml_count} YAML files, found {len(self.data_by_symbol)} symbols")
        return self.data_by_symbol
    
    def save_to_csv(self):
        fieldnames = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'sector']
        
        for symbol, rows in self.data_by_symbol.items():
            if not rows:
                continue
                
            rows_sorted = sorted(rows, key=lambda r: r['date'])
            output_path = os.path.join(self.output_dir, f"{symbol}.csv")
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_sorted)

        print(f"Exported {len(self.data_by_symbol)} symbol CSVs")
    
    def create_combined_dataset(self):
        all_data = []
        for symbol, rows in self.data_by_symbol.items():
            all_data.extend(rows)
        
        if not all_data:
            print("No data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        if df.empty or 'date' not in df.columns or 'ticker' not in df.columns:
            print("Invalid DataFrame")
            return df
        
        try:
            df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        except Exception as e:
            print(f"Error sorting DataFrame: {str(e)}")
            return df
        
        combined_path = os.path.join(self.output_dir, "nifty50_combined.csv")
        df.to_csv(combined_path, index=False)
        
        print(f"Created combined dataset: {len(df)} records")
        return df
    
    def run_extraction(self):
        print("Starting data extraction...")
        self.extract_data()
        self.save_to_csv()
        combined_df = self.create_combined_dataset()
        
        if not combined_df.empty:
            print(f"Extraction complete: {len(combined_df)} records, {combined_df['ticker'].nunique()} stocks")
        
        return combined_df

def main():
    extractor = NiftyDataExtractor()
    return extractor.run_extraction()

if __name__ == "__main__":
    main()