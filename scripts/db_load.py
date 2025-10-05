import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Date, Numeric, BigInteger, Integer
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

class Symbols(Base):
    __tablename__ = 'symbols'
    symbol = Column(String(16), primary_key=True)
    name = Column(String(255))
    sector = Column(String(128))

class DailyPrices(Base):
    __tablename__ = 'daily_prices'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(16), nullable=False)
    trade_date = Column(Date, nullable=False)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    volume = Column(BigInteger)
    
    __table_args__ = (
        sa.UniqueConstraint('symbol', 'trade_date', name='unique_symbol_date'),
    )

class YearlyMetrics(Base):
    __tablename__ = 'yearly_metrics'
    symbol = Column(String(16), primary_key=True)
    year = Column(Integer, primary_key=True)
    start_price = Column(Numeric)
    end_price = Column(Numeric)
    yearly_return = Column(Numeric)
    volatility = Column(Numeric)
    avg_volume = Column(Numeric)

class DatabaseManager:
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = "postgresql://username:password@localhost:5432/your_database"
        
        self.db_url = db_url
        self.engine = create_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized: {db_url}")
    
    def load_symbols(self, df):
        session = self.Session()
        try:
            session.query(Symbols).delete()
            
            symbols_data = df.groupby('ticker').agg({'sector': 'first'}).reset_index()
            
            for _, row in symbols_data.iterrows():
                symbol_record = Symbols(
                    symbol=row['ticker'],
                    name=row['ticker'],
                    sector=row['sector']
                )
                session.add(symbol_record)
            
            session.commit()
            logger.info("Symbols loaded successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading symbols: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_daily_prices(self, df, batch_size=1000):
        session = self.Session()
        try:
            session.query(DailyPrices).delete()
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    price_record = DailyPrices(
                        symbol=row['ticker'],
                        trade_date=row['date'].date(),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=int(row['volume'])
                    )
                    session.add(price_record)
                
                session.commit()
                logger.info(f"Loaded batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            logger.info("Daily prices loaded successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading daily prices: {str(e)}")
            raise
        finally:
            session.close()
    
    def load_yearly_metrics(self, df):
        session = self.Session()
        try:
            session.query(YearlyMetrics).delete()
            
            current_year = datetime.now().year
            
            for _, row in df.iterrows():
                metrics_record = YearlyMetrics(
                    symbol=row['ticker'],
                    year=current_year,
                    start_price=row.get('start_price', 0),
                    end_price=row.get('end_price', 0),
                    yearly_return=row['yearly_return'],
                    volatility=row.get('avg_volatility', 0),
                    avg_volume=row['avg_volume']
                )
                session.add(metrics_record)
            
            session.commit()
            logger.info("Yearly metrics loaded successfully")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading yearly metrics: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_top_performers(self, limit=10, ascending=False):
        session = self.Session()
        try:
            query = session.query(YearlyMetrics).order_by(
                YearlyMetrics.yearly_return.desc() if not ascending else YearlyMetrics.yearly_return.asc()
            ).limit(limit)
            
            results = query.all()
            return [{
                'symbol': r.symbol,
                'yearly_return': float(r.yearly_return),
                'volatility': float(r.volatility),
                'avg_volume': float(r.avg_volume)
            } for r in results]
            
        except Exception as e:
            logger.error(f"Error getting top performers: {str(e)}")
            return []
        finally:
            session.close()
    
    def run_database_setup(self, stock_data_path, metrics_data_path):
        logger.info("Starting database setup...")
        
        try:
            if os.path.exists(stock_data_path):
                stock_df = pd.read_csv(stock_data_path)
                stock_df['date'] = pd.to_datetime(stock_df['date'])
                
                self.load_symbols(stock_df)
                self.load_daily_prices(stock_df)
            else:
                logger.warning(f"Stock data file not found: {stock_data_path}")
            
            if os.path.exists(metrics_data_path):
                metrics_df = pd.read_csv(metrics_data_path)
                self.load_yearly_metrics(metrics_df)
            else:
                logger.warning(f"Metrics data file not found: {metrics_data_path}")
            
            logger.info("Database setup completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in database setup: {str(e)}")
            raise

def main():
    # Update these credentials with your actual PostgreSQL details
    db_url = "postgresql://postgres:1998@localhost:5432/postgres"
    db_manager = DatabaseManager(db_url)
    
    stock_data_path = r"nifty50_cleaned.csv"
    metrics_data_path = r"nifty50_yearly_metrics.csv"
    
    db_manager.run_database_setup(stock_data_path, metrics_data_path)
    
    top_performers = db_manager.get_top_performers(10)
    logger.info("Top 10 Performers:")
    for i, stock in enumerate(top_performers, 1):
        logger.info(f"{i}. {stock['symbol']}: {stock['yearly_return']:.2f}%")

if __name__ == "__main__":
    main()