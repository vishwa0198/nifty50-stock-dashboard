import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class NiftyVisualizer:
    def __init__(self, data_path=r"C:\nifty50-stock-dashboard\nifty50_cleaned.csv"):
        self.data_path = data_path
        self.df = None
        self.yearly_metrics = None
        self.load_data()
        
        self.colors = {
            'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c',
            'danger': '#d62728', 'warning': '#ff7f0e', 'info': '#17a2b8'
        }
    
    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"Loaded {len(self.df)} records for visualization")
        else:
            print(f"Data file not found: {self.data_path}")
            return
        
        yearly_path = r"C:\nifty50-stock-dashboard\nifty50_yearly_metrics.csv"
        if os.path.exists(yearly_path):
            self.yearly_metrics = pd.read_csv(yearly_path)
            print(f"Loaded yearly metrics for visualization")
    
    def create_top_performers_chart(self, top_n=10, ascending=False):
        if self.yearly_metrics is None:
            return go.Figure()
        
        if ascending:
            data = self.yearly_metrics.nsmallest(top_n, 'yearly_return')
            title = f"Top {top_n} Worst Performing Stocks"
            color = self.colors['danger']
        else:
            data = self.yearly_metrics.nlargest(top_n, 'yearly_return')
            title = f"Top {top_n} Best Performing Stocks"
            color = self.colors['success']
        
        fig = go.Figure(data=[
            go.Bar(
                x=data['ticker'],
                y=data['yearly_return'],
                text=[f"{val:.2f}%" for val in data['yearly_return']],
                textposition='auto',
                marker_color=color,
                hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<br>Sector: %{customdata}<extra></extra>',
                customdata=data['sector']
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Stock Ticker",
            yaxis_title="Yearly Return (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_volatility_chart(self, top_n=10):
        if self.df is None:
            return go.Figure()
        
        volatility_data = []
        for ticker in self.df['ticker'].unique():
            stock_data = self.df[self.df['ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('date')
            
            if len(stock_data) < 2:
                continue
            
            stock_data['daily_return'] = stock_data['close'].pct_change()
            volatility = stock_data['daily_return'].std() * np.sqrt(252)
            
            volatility_data.append({
                'ticker': ticker,
                'sector': stock_data['sector'].iloc[0],
                'volatility': volatility
            })
        
        volatility_df = pd.DataFrame(volatility_data)
        volatility_df = volatility_df.sort_values('volatility', ascending=False).head(top_n)
        
        fig = go.Figure(data=[
            go.Bar(
                x=volatility_df['ticker'],
                y=volatility_df['volatility'],
                text=[f"{val:.2f}" for val in volatility_df['volatility']],
                textposition='auto',
                marker_color=self.colors['warning'],
                hovertemplate='<b>%{x}</b><br>Volatility: %{y:.2f}<br>Sector: %{customdata}<extra></extra>',
                customdata=volatility_df['sector']
            )
        ])
        
        fig.update_layout(
            title=f"Top {top_n} Most Volatile Stocks",
            xaxis_title="Stock Ticker",
            yaxis_title="Volatility (Annualized)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_cumulative_returns_chart(self, top_n=5):
        if self.df is None:
            return go.Figure()
        
        cumulative_returns_data = []
        
        for ticker in self.df['ticker'].unique():
            stock_data = self.df[self.df['ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('date')
            
            if len(stock_data) < 2:
                continue
            
            stock_data['daily_return'] = stock_data['close'].pct_change()
            stock_data['cumulative_return'] = (1 + stock_data['daily_return']).cumprod() - 1
            
            final_return = stock_data['cumulative_return'].iloc[-1]
            
            cumulative_returns_data.append({
                'ticker': ticker,
                'final_return': final_return,
                'data': stock_data[['date', 'cumulative_return']].copy()
            })
        
        cumulative_returns_data.sort(key=lambda x: x['final_return'], reverse=True)
        top_stocks = cumulative_returns_data[:top_n]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1[:top_n]
        
        for i, stock in enumerate(top_stocks):
            fig.add_trace(go.Scatter(
                x=stock['data']['date'],
                y=stock['data']['cumulative_return'] * 100,
                mode='lines',
                name=stock['ticker'],
                line=dict(color=colors[i], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Cumulative Return: %{y:.2f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Cumulative Returns - Top {top_n} Performing Stocks",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_sector_performance_chart(self):
        if self.yearly_metrics is None:
            return go.Figure()
        
        sector_analysis = self.yearly_metrics.groupby('sector').agg({
            'yearly_return': 'mean',
            'ticker': 'count'
        }).reset_index()
        sector_analysis.columns = ['sector', 'avg_return', 'stock_count']
        sector_analysis = sector_analysis.sort_values('avg_return', ascending=False)
        
        colors = [self.colors['success'] if x > 0 else self.colors['danger'] for x in sector_analysis['avg_return']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sector_analysis['sector'],
                y=sector_analysis['avg_return'],
                text=[f"{val:.2f}%" for val in sector_analysis['avg_return']],
                textposition='auto',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Avg Return: %{y:.2f}%<br>Stocks: %{customdata}<extra></extra>',
                customdata=sector_analysis['stock_count']
            )
        ])
        
        fig.update_layout(
            title="Average Yearly Return by Sector",
            xaxis_title="Sector",
            yaxis_title="Average Yearly Return (%)",
            template="plotly_white",
            height=500,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_correlation_heatmap(self, sample_size=20):
        if self.df is None:
            return go.Figure()
        
        sample_stocks = self.df['ticker'].unique()[:sample_size]
        sample_data = self.df[self.df['ticker'].isin(sample_stocks)]
        
        price_pivot = sample_data.pivot_table(
            index='date', 
            columns='ticker', 
            values='close', 
            aggfunc='first'
        )
        
        price_changes = price_pivot.pct_change().dropna()
        correlation_matrix = price_changes.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Stock Price Correlation Heatmap (Sample of {sample_size} stocks)",
            template="plotly_white",
            height=600,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_market_overview_dashboard(self):
        if self.yearly_metrics is None:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Top 10 Gainers",
                "Top 10 Losers", 
                "Sector Performance",
                "Market Distribution"
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        top_gainers = self.yearly_metrics.nlargest(10, 'yearly_return')
        fig.add_trace(
            go.Bar(
                x=top_gainers['ticker'],
                y=top_gainers['yearly_return'],
                marker_color=self.colors['success'],
                name="Gainers",
                showlegend=False
            ),
            row=1, col=1
        )
        
        top_losers = self.yearly_metrics.nsmallest(10, 'yearly_return')
        fig.add_trace(
            go.Bar(
                x=top_losers['ticker'],
                y=top_losers['yearly_return'],
                marker_color=self.colors['danger'],
                name="Losers",
                showlegend=False
            ),
            row=1, col=2
        )
        
        sector_analysis = self.yearly_metrics.groupby('sector')['yearly_return'].mean().reset_index()
        sector_analysis = sector_analysis.sort_values('yearly_return', ascending=False)
        
        colors_sector = [self.colors['success'] if x > 0 else self.colors['danger'] for x in sector_analysis['yearly_return']]
        
        fig.add_trace(
            go.Bar(
                x=sector_analysis['sector'],
                y=sector_analysis['yearly_return'],
                marker_color=colors_sector,
                name="Sector Performance",
                showlegend=False
            ),
            row=2, col=1
        )
        
        green_count = (self.yearly_metrics['yearly_return'] > 0).sum()
        red_count = (self.yearly_metrics['yearly_return'] <= 0).sum()
        
        fig.add_trace(
            go.Pie(
                labels=['Green Stocks', 'Red Stocks'],
                values=[green_count, red_count],
                marker_colors=[self.colors['success'], self.colors['danger']],
                name="Market Distribution",
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Nifty 50 Market Overview Dashboard",
            template="plotly_white",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_risk_return_scatter(self):
        if self.df is None or self.yearly_metrics is None:
            return go.Figure()
        
        risk_return_data = []
        
        for ticker in self.df['ticker'].unique():
            stock_data = self.df[self.df['ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('date')
            
            if len(stock_data) < 2:
                continue
            
            stock_data['daily_return'] = stock_data['close'].pct_change()
            volatility = stock_data['daily_return'].std() * np.sqrt(252)
            
            yearly_return = self.yearly_metrics[self.yearly_metrics['ticker'] == ticker]['yearly_return'].iloc[0]
            sector = stock_data['sector'].iloc[0]
            
            risk_return_data.append({
                'ticker': ticker,
                'volatility': volatility,
                'yearly_return': yearly_return,
                'sector': sector
            })
        
        risk_return_df = pd.DataFrame(risk_return_data)
        
        fig = px.scatter(
            risk_return_df,
            x='volatility',
            y='yearly_return',
            color='sector',
            hover_data=['ticker'],
            title="Risk vs Return Analysis",
            labels={
                'volatility': 'Volatility (Risk)',
                'yearly_return': 'Yearly Return (%)'
            }
        )
        
        fig.update_layout(
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def save_all_charts(self, output_dir="charts"):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating and saving all charts...")
        
        charts = {
            'top_gainers': self.create_top_performers_chart(10, False),
            'top_losers': self.create_top_performers_chart(10, True),
            'volatility': self.create_volatility_chart(10),
            'cumulative_returns': self.create_cumulative_returns_chart(5),
            'sector_performance': self.create_sector_performance_chart(),
            'correlation_heatmap': self.create_correlation_heatmap(20),
            'market_overview': self.create_market_overview_dashboard(),
            'risk_return': self.create_risk_return_scatter()
        }
        
        for name, fig in charts.items():
            if fig.data:
                fig.write_html(f"{output_dir}/{name}.html")
                print(f"Saved {name} chart")
        
        print(f"All charts saved to {output_dir}")

def main():
    visualizer = NiftyVisualizer()
    
    if visualizer.df is None:
        print("No data available for visualization")
        return
    
    visualizer.save_all_charts()
    
    print("Visualization creation completed successfully!")

if __name__ == "__main__":
    main()