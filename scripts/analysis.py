import pandas as pd
import numpy as np
from datetime import datetime
import os

class NiftyAnalyzer:
    def __init__(self, data_path=r"C:\nifty50-stock-dashboard\nifty50_cleaned.csv"):
        self.data_path = data_path
        self.df = None
        self.yearly_metrics = None
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"Loaded {len(self.df)} records for {self.df['ticker'].nunique()} stocks")
        else:
            print(f"Data file not found: {self.data_path}")
            return
        
        yearly_path = r"C:\nifty50-stock-dashboard\nifty50_yearly_metrics.csv"
        if os.path.exists(yearly_path):
            self.yearly_metrics = pd.read_csv(yearly_path)
            print(f"Loaded yearly metrics for {len(self.yearly_metrics)} stocks")
    
    def get_top_performers(self, n=10, ascending=False):
        if self.yearly_metrics is None:
            return pd.DataFrame()
        
        return self.yearly_metrics.nlargest(n, 'yearly_return') if not ascending else self.yearly_metrics.nsmallest(n, 'yearly_return')
    
    def get_market_summary(self):
        if self.df is None:
            return {}
        
        total_stocks = self.df['ticker'].nunique()
        
        if self.yearly_metrics is not None:
            green_stocks = (self.yearly_metrics['yearly_return'] > 0).sum()
            red_stocks = (self.yearly_metrics['yearly_return'] <= 0).sum()
        else:
            yearly_returns = self.df.groupby('ticker').apply(
                lambda x: ((x['close'].iloc[-1] - x['close'].iloc[0]) / x['close'].iloc[0]) * 100
            )
            green_stocks = (yearly_returns > 0).sum()
            red_stocks = (yearly_returns <= 0).sum()
        
        avg_price = self.df['close'].mean()
        avg_volume = self.df['volume'].mean()
        
        return {
            'total_stocks': total_stocks,
            'green_stocks': green_stocks,
            'red_stocks': red_stocks,
            'green_percentage': (green_stocks / total_stocks) * 100,
            'red_percentage': (red_stocks / total_stocks) * 100,
            'avg_price': avg_price,
            'avg_volume': avg_volume,
            'date_range': (self.df['date'].min(), self.df['date'].max())
        }
    
    def analyze_volatility(self, top_n=10):
        if self.df is None:
            return pd.DataFrame()
        
        volatility_data = []
        
        for ticker in self.df['ticker'].unique():
            stock_data = self.df[self.df['ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('date')
            
            if len(stock_data) < 2:
                continue
            
            stock_data['daily_return'] = stock_data['close'].pct_change()
            volatility = stock_data['daily_return'].std() * np.sqrt(252)
            avg_return = stock_data['daily_return'].mean() * 252
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            cumulative_returns = (1 + stock_data['daily_return']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            volatility_data.append({
                'ticker': ticker,
                'sector': stock_data['sector'].iloc[0],
                'volatility': volatility,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trading_days': len(stock_data)
            })
        
        volatility_df = pd.DataFrame(volatility_data)
        volatility_df = volatility_df.sort_values('volatility', ascending=False)
        
        return volatility_df.head(top_n)
    
    def analyze_sector_performance(self):
        if self.yearly_metrics is None:
            return pd.DataFrame()
        
        sector_analysis = self.yearly_metrics.groupby('sector').agg({
            'yearly_return': ['mean', 'std', 'min', 'max', 'count'],
            'avg_volatility': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean'
        }).round(2)
        
        sector_analysis.columns = ['_'.join(col).strip() for col in sector_analysis.columns]
        sector_analysis = sector_analysis.reset_index()
        
        sector_analysis.columns = [
            'sector', 'avg_return', 'return_std', 'min_return', 'max_return', 
            'stock_count', 'avg_volatility', 'avg_sharpe', 'avg_drawdown'
        ]
        
        sector_analysis = sector_analysis.sort_values('avg_return', ascending=False)
        return sector_analysis
    
    def analyze_correlation(self):
        if self.df is None:
            return pd.DataFrame()
        
        price_pivot = self.df.pivot_table(
            index='date', 
            columns='ticker', 
            values='close', 
            aggfunc='first'
        )
        
        price_changes = price_pivot.pct_change().dropna()
        correlation_matrix = price_changes.corr()
        
        return correlation_matrix
    
    def analyze_monthly_performance(self):
        if self.df is None:
            return {}
        
        self.df['year_month'] = self.df['date'].dt.to_period('M')
        
        monthly_performance = {}
        
        for period in self.df['year_month'].unique():
            period_data = self.df[self.df['year_month'] == period]
            
            monthly_returns = []
            for ticker in period_data['ticker'].unique():
                ticker_data = period_data[period_data['ticker'] == ticker].sort_values('date')
                
                if len(ticker_data) >= 2:
                    first_price = ticker_data['close'].iloc[0]
                    last_price = ticker_data['close'].iloc[-1]
                    monthly_return = ((last_price - first_price) / first_price) * 100
                    
                    monthly_returns.append({
                        'ticker': ticker,
                        'sector': ticker_data['sector'].iloc[0],
                        'monthly_return': monthly_return
                    })
            
            monthly_df = pd.DataFrame(monthly_returns)
            monthly_df = monthly_df.sort_values('monthly_return', ascending=False)
            
            top_gainers = monthly_df.head(5)
            top_losers = monthly_df.tail(5)
            
            monthly_performance[str(period)] = {
                'top_gainers': top_gainers.to_dict('records'),
                'top_losers': top_losers.to_dict('records'),
                'avg_return': monthly_df['monthly_return'].mean(),
                'total_stocks': len(monthly_df)
            }
        
        return monthly_performance
    
    def calculate_risk_metrics(self):
        if self.df is None:
            return pd.DataFrame()
        
        risk_metrics = []
        
        for ticker in self.df['ticker'].unique():
            stock_data = self.df[self.df['ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('date')
            
            if len(stock_data) < 30:
                continue
            
            stock_data['daily_return'] = stock_data['close'].pct_change()
            returns = stock_data['daily_return'].dropna()
            
            if len(returns) < 20:
                continue
            
            volatility = returns.std() * np.sqrt(252)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            risk_free_rate = 0.05
            excess_return = returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
            
            risk_metrics.append({
                'ticker': ticker,
                'sector': stock_data['sector'].iloc[0],
                'volatility': volatility,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            })
        
        risk_df = pd.DataFrame(risk_metrics)
        risk_df = risk_df.sort_values('volatility', ascending=False)
        
        return risk_df
    
    def generate_insights(self):
        print("Generating market insights...")
        
        insights = {}
        insights['market_summary'] = self.get_market_summary()
        insights['top_gainers'] = self.get_top_performers(10, ascending=False)
        insights['top_losers'] = self.get_top_performers(10, ascending=True)
        insights['most_volatile'] = self.analyze_volatility(10)
        insights['sector_performance'] = self.analyze_sector_performance()
        insights['risk_metrics'] = self.calculate_risk_metrics()
        insights['monthly_performance'] = self.analyze_monthly_performance()
        
        print("Market insights generated successfully")
        return insights
    
    def save_analysis_results(self, output_dir="analysis_results"):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Saving analysis results...")
        
        insights = self.generate_insights()
        
        if not insights['top_gainers'].empty:
            insights['top_gainers'].to_csv(f"{output_dir}/top_gainers.csv", index=False)
        
        if not insights['top_losers'].empty:
            insights['top_losers'].to_csv(f"{output_dir}/top_losers.csv", index=False)
        
        if not insights['most_volatile'].empty:
            insights['most_volatile'].to_csv(f"{output_dir}/volatility_analysis.csv", index=False)
        
        if not insights['sector_performance'].empty:
            insights['sector_performance'].to_csv(f"{output_dir}/sector_performance.csv", index=False)
        
        if not insights['risk_metrics'].empty:
            insights['risk_metrics'].to_csv(f"{output_dir}/risk_metrics.csv", index=False)
        
        correlation_matrix = self.analyze_correlation()
        if not correlation_matrix.empty:
            correlation_matrix.to_csv(f"{output_dir}/correlation_matrix.csv")
        
        import json
        with open(f"{output_dir}/market_summary.json", 'w') as f:
            json.dump(insights['market_summary'], f, indent=2, default=str)
        
        with open(f"{output_dir}/monthly_performance.json", 'w') as f:
            json.dump(insights['monthly_performance'], f, indent=2, default=str)
        
        print(f"Analysis results saved to {output_dir}")

def main():
    analyzer = NiftyAnalyzer()
    
    if analyzer.df is None:
        print("No data available for analysis")
        return
    
    insights = analyzer.generate_insights()
    
    print("\n" + "="*60)
    print("NIFTY 50 ANALYSIS INSIGHTS")
    print("="*60)
    
    market_summary = insights['market_summary']
    print(f"Total Stocks: {market_summary['total_stocks']}")
    print(f"Green Stocks: {market_summary['green_stocks']} ({market_summary['green_percentage']:.1f}%)")
    print(f"Red Stocks: {market_summary['red_stocks']} ({market_summary['red_percentage']:.1f}%)")
    print(f"Average Price: ₹{market_summary['avg_price']:.2f}")
    
    print("\nTop 5 Gainers:")
    for i, (_, stock) in enumerate(insights['top_gainers'].head(5).iterrows(), 1):
        print(f"{i}. {stock['ticker']}: {stock['yearly_return']:.2f}%")
    
    print("\nTop 5 Losers:")
    for i, (_, stock) in enumerate(insights['top_losers'].head(5).iterrows(), 1):
        print(f"{i}. {stock['ticker']}: {stock['yearly_return']:.2f}%")
    
    print("\nTop 3 Sectors by Performance:")
    for i, (_, sector) in enumerate(insights['sector_performance'].head(3).iterrows(), 1):
        print(f"{i}. {sector['sector']}: {sector['avg_return']:.2f}% avg return")
    
    analyzer.save_analysis_results()
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()