import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

sys.path.append('scripts')

try:
    from scripts.analysis import NiftyAnalyzer
    from scripts.visualization import NiftyVisualizer
    from scripts.powerbi import PowerBIDataExporter
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

st.set_page_config(
    page_title="Nifty 50 Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        analyzer = NiftyAnalyzer()
        
        if analyzer.df is None:
            return None, None, None
        
        insights = analyzer.generate_insights()
        
        return analyzer.df, analyzer.yearly_metrics, insights
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def get_market_summary(insights):
    if insights is None:
        return {}
    return insights.get('market_summary', {})

def main():
    st.markdown('<h1 class="main-header">📈 Nifty 50 Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    with st.spinner("Loading data..."):
        df, yearly_metrics, insights = load_data()
    
    if df is None:
        st.error("No data available. Please ensure the data files are present.")
        st.stop()
    
    st.sidebar.title("📊 Dashboard Controls")
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    sectors = ['All'] + sorted(df['sector'].unique().tolist())
    selected_sector = st.sidebar.selectbox("Select Sector", sectors)
    
    if selected_sector != 'All':
        available_stocks = df[df['sector'] == selected_sector]['ticker'].unique()
    else:
        available_stocks = df['ticker'].unique()
    
    selected_stock = st.sidebar.selectbox("Select Individual Stock", ['None'] + sorted(available_stocks.tolist()))
    st.sidebar.markdown("---")
    if st.sidebar.button("Export data for Power BI"):
        try:
            exporter = PowerBIDataExporter()
            exporter.export_for_powerbi()
            st.sidebar.success("Exported to powerbi_data/*.csv")
        except Exception as e:
            st.sidebar.error(f"Power BI export failed: {e}")
    
    filtered_df = df.copy()
    if selected_sector != 'All':
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) & 
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Overview", 
        "📈 Performance Analysis", 
        "🔍 Risk Analysis", 
        "📅 Monthly Analysis",
        "🎯 Individual Stock",
        "🧩 Power BI"
    ])
    
    with tab1:
        st.header("📊 Market Overview")
        
        market_summary = get_market_summary(insights)
        if market_summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Stocks",
                    market_summary.get('total_stocks', 0),
                    help="Total number of stocks in the dataset"
                )
            
            with col2:
                st.metric(
                    "Green Stocks",
                    f"{market_summary.get('green_stocks', 0)} ({market_summary.get('green_percentage', 0):.1f}%)",
                    help="Stocks with positive yearly returns"
                )
            
            with col3:
                st.metric(
                    "Red Stocks",
                    f"{market_summary.get('red_stocks', 0)} ({market_summary.get('red_percentage', 0):.1f}%)",
                    help="Stocks with negative yearly returns"
                )
            
            with col4:
                st.metric(
                    "Avg Price",
                    f"₹{market_summary.get('avg_price', 0):.2f}",
                    help="Average closing price across all stocks"
                )
        
        if insights:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Gainers")
                if not insights['top_gainers'].empty:
                    fig_gainers = px.bar(
                        insights['top_gainers'].head(10),
                        x='ticker',
                        y='yearly_return',
                        title="Top 10 Performing Stocks",
                        color='yearly_return',
                        color_continuous_scale='Greens'
                    )
                    fig_gainers.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_gainers, use_container_width=True)
            
            with col2:
                st.subheader("Top 10 Losers")
                if not insights['top_losers'].empty:
                    fig_losers = px.bar(
                        insights['top_losers'].head(10),
                        x='ticker',
                        y='yearly_return',
                        title="Worst 10 Performing Stocks",
                        color='yearly_return',
                        color_continuous_scale='Reds'
                    )
                    fig_losers.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_losers, use_container_width=True)
            
            st.subheader("Sector Performance")
            if not insights['sector_performance'].empty:
                fig_sector = px.bar(
                    insights['sector_performance'],
                    x='sector',
                    y='avg_return',
                    title="Average Yearly Return by Sector",
                    color='avg_return',
                    color_continuous_scale='RdYlGn'
                )
                fig_sector.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_sector, use_container_width=True)
    
    with tab2:
        st.header("📈 Performance Analysis")
        
        st.subheader("Cumulative Returns Over Time")
        
        if yearly_metrics is not None:
            top_5_tickers = yearly_metrics.nlargest(5, 'yearly_return')['ticker'].tolist()
            
            fig_cumulative = go.Figure()
            
            colors = px.colors.qualitative.Set1[:5]
            
            for i, ticker in enumerate(top_5_tickers):
                stock_data = filtered_df[filtered_df['ticker'] == ticker].sort_values('date')
                if len(stock_data) > 1:
                    stock_data = stock_data.copy()
                    stock_data['daily_return'] = stock_data['close'].pct_change()
                    stock_data['cumulative_return'] = (1 + stock_data['daily_return']).cumprod() - 1
                    
                    fig_cumulative.add_trace(go.Scatter(
                        x=stock_data['date'],
                        y=stock_data['cumulative_return'] * 100,
                        mode='lines',
                        name=ticker,
                        line=dict(color=colors[i], width=2)
                    ))
            
            fig_cumulative.update_layout(
                title="Cumulative Returns - Top 5 Performers",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        st.subheader("Detailed Performance Metrics")
        if yearly_metrics is not None:
            display_metrics = yearly_metrics[['ticker', 'sector', 'yearly_return', 'avg_volatility', 'sharpe_ratio']].copy()
            display_metrics = display_metrics.round(2)
            display_metrics.columns = ['Ticker', 'Sector', 'Yearly Return (%)', 'Volatility', 'Sharpe Ratio']
            
            st.dataframe(display_metrics, use_container_width=True)
            
            csv = display_metrics.to_csv(index=False)
            st.download_button(
                label="Download Performance Data",
                data=csv,
                file_name="nifty50_performance.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("🔍 Risk Analysis")
        
        st.subheader("Most Volatile Stocks")
        if insights and not insights['most_volatile'].empty:
            fig_volatility = px.bar(
                insights['most_volatile'].head(10),
                x='ticker',
                y='volatility',
                title="Top 10 Most Volatile Stocks",
                color='volatility',
                color_continuous_scale='Oranges'
            )
            fig_volatility.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_volatility, use_container_width=True)
        
        st.subheader("Risk Metrics Summary")
        if insights and not insights['risk_metrics'].empty:
            risk_display = insights['risk_metrics'][['ticker', 'sector', 'volatility', 'max_drawdown', 'sharpe_ratio']].head(10)
            risk_display = risk_display.round(3)
            risk_display.columns = ['Ticker', 'Sector', 'Volatility', 'Max Drawdown', 'Sharpe Ratio']
            
            st.dataframe(risk_display, use_container_width=True)
        
        st.subheader("Risk vs Return Analysis")
        if yearly_metrics is not None and insights and not insights['most_volatile'].empty:
            scatter_data = yearly_metrics.merge(
                insights['most_volatile'][['ticker', 'volatility']], 
                on='ticker', 
                how='inner'
            )
            
            fig_scatter = px.scatter(
                scatter_data,
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
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Correlation Heatmap")
        try:
            # Build correlation on filtered data (limit to avoid overcrowding)
            sample_size = min(30, len(filtered_df['ticker'].unique()))
            sample_tickers = sorted(filtered_df['ticker'].unique())[:sample_size]
            corr_df = filtered_df[filtered_df['ticker'].isin(sample_tickers)]
            price_pivot = corr_df.pivot_table(index='date', columns='ticker', values='close', aggfunc='first')
            price_changes = price_pivot.pct_change().dropna()
            correlation_matrix = price_changes.corr()
            if not correlation_matrix.empty:
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=False,
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    title=f"Stock Price Correlation Heatmap (n={correlation_matrix.shape[0]})"
                )
                fig_corr.update_layout(height=600, xaxis_tickangle=-45)
                st.plotly_chart(fig_corr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate correlation heatmap: {e}")
    
    with tab4:
        st.header("📅 Monthly Analysis")
        
        st.subheader("Monthly Performance Trends")
        if insights and 'monthly_performance' in insights:
            monthly_data = insights['monthly_performance']
            
            monthly_summary = []
            for month, data in monthly_data.items():
                monthly_summary.append({
                    'Month': month,
                    'Avg Return': data['avg_return'],
                    'Total Stocks': data['total_stocks']
                })
            
            if monthly_summary:
                monthly_df = pd.DataFrame(monthly_summary)
                
                fig_monthly = px.line(
                    monthly_df,
                    x='Month',
                    y='Avg Return',
                    title="Average Monthly Returns",
                    markers=True
                )
                fig_monthly.update_layout(height=400)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                st.subheader("Monthly Performance Summary")
                st.dataframe(monthly_df, use_container_width=True)

                # Top 5 Gainers and Losers (Month-wise)
                st.subheader("Top 5 Gainers and Losers by Month")
                months_sorted = sorted(monthly_data.keys())
                selected_month = st.selectbox("Select Month", months_sorted, index=len(months_sorted)-1)
                month_payload = monthly_data.get(selected_month, {})
                gainers = pd.DataFrame(month_payload.get('top_gainers', []))
                losers = pd.DataFrame(month_payload.get('top_losers', []))

                col_g, col_l = st.columns(2)
                with col_g:
                    st.write(f"Top 5 Gainers - {selected_month}")
                    if not gainers.empty:
                        fig_g = px.bar(
                            gainers.sort_values('monthly_return', ascending=False),
                            x='ticker', y='monthly_return', color='monthly_return',
                            title=None, color_continuous_scale='Greens'
                        )
                        fig_g.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_g, use_container_width=True)
                    else:
                        st.info("No data for gainers in selected month")

                with col_l:
                    st.write(f"Top 5 Losers - {selected_month}")
                    if not losers.empty:
                        fig_l = px.bar(
                            losers.sort_values('monthly_return', ascending=True),
                            x='ticker', y='monthly_return', color='monthly_return',
                            title=None, color_continuous_scale='Reds'
                        )
                        fig_l.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig_l, use_container_width=True)
                    else:
                        st.info("No data for losers in selected month")
    
    with tab5:
        st.header(f"🎯 Individual Stock Analysis - {selected_stock}")
        
        if selected_stock != 'None':
            stock_data = filtered_df[filtered_df['ticker'] == selected_stock].sort_values('date')
            
            if not stock_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = stock_data['close'].iloc[-1]
                    st.metric("Current Price", f"₹{current_price:.2f}")
                
                with col2:
                    price_change = ((current_price - stock_data['close'].iloc[0]) / stock_data['close'].iloc[0]) * 100
                    st.metric("Total Return", f"{price_change:.2f}%")
                
                with col3:
                    avg_volume = stock_data['volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
                
                with col4:
                    volatility = stock_data['close'].pct_change().std() * np.sqrt(252)
                    st.metric("Volatility", f"{volatility:.2f}")
                
                st.subheader("Price History")
                fig_price = go.Figure()
                
                fig_price.add_trace(go.Scatter(
                    x=stock_data['date'],
                    y=stock_data['close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig_price.update_layout(
                    title=f"{selected_stock} - Price History",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
                
                st.subheader("Volume Analysis")
                fig_volume = px.bar(
                    stock_data,
                    x='date',
                    y='volume',
                    title=f"{selected_stock} - Trading Volume"
                )
                fig_volume.update_layout(height=300)
                st.plotly_chart(fig_volume, use_container_width=True)
                
                st.subheader("Stock Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Price Statistics**")
                    price_stats = stock_data['close'].describe()
                    st.write(price_stats)
                
                with col2:
                    st.write("**Volume Statistics**")
                    volume_stats = stock_data['volume'].describe()
                    st.write(volume_stats)
            else:
                st.warning(f"No data available for {selected_stock} in the selected date range.")
        else:
            st.info("Please select a stock from the sidebar to view individual analysis.")

    with tab6:
        st.header("🧩 Power BI Report")
        st.info("Use the sidebar button 'Export data for Power BI' to refresh CSVs. Then open your Power BI report connected to the powerbi_data folder. If you have a published report link, paste it below to embed.")
        pbi_url = st.text_input("Power BI report 'Embed' URL (public or organizational)")
        if pbi_url:
            # Simple iframe embed
            st.components.v1.iframe(src=pbi_url, height=700, scrolling=True)
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>📊 Nifty 50 Stock Analysis Dashboard | Built with Streamlit & Plotly</p>
            <p>Data updated: {}</p>
        </div>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()