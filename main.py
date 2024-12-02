# Core analysis modules
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional

# Technical analysis
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Financial metrics and risk
from scipy import stats
import empyrical as ep

#import sub files
from helper_compute_index import BasePortfolioAnalyzer
from helper_generate_report import ReportGeneratorBase

class EnhancedPortfolioAnalyzer(BasePortfolioAnalyzer, ReportGeneratorBase):
    def __init__(self, config: dict):
        """
        Initialize enhanced portfolio analyzer with configuration
        
        Parameters:
            config (dict): Configuration dictionary containing:
                - tickers: List of asset tickers
                - weights: Portfolio weights
                - risk_free_rate: Risk-free rate for calculations
                - benchmark: Benchmark ticker (e.g., 'SPY')
        """
        super().__init__()
        self.config = config
        self.tickers = config['tickers']
        self.weights = np.array(config.get('weights', [1/len(self.tickers)] * len(self.tickers)))
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.benchmark = config.get('benchmark', 'SPY')
        
        # Initialize data containers
        self.price_data = None
        self.returns_data = None
        self.benchmark_data = None
        self.technical_indicators = None
        self.fundamentals = None
        
    
    def fetch_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        获取市场数据，包括价格、交易量和基本面数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            包含所有股票数据的字典
        """
        # 初始化数据字典
        data = {}
        # 初始化基本面数据字典（避免 NoneType 错误）
        self.fundamentals = {}
        
        # 获取所有股票（包括基准股票）的数据
        for ticker in self.tickers + [self.benchmark]:
            try:
                # 创建股票对象
                stock = yf.Ticker(ticker)
                
                # 获取历史价格数据
                ticker_data = stock.history(start=start_date, end=end_date)
                data[ticker] = ticker_data
                
                # 获取基本面数据
                info = stock.info
                self.fundamentals[ticker] = {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('forwardPE'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta')
                }
                
            except Exception as e:
                print(f"获取{ticker}数据时出错: {str(e)}")
                # 如果获取数据失败，添加空数据防止后续处理出错
                data[ticker] = pd.DataFrame()
                self.fundamentals[ticker] = {
                    'market_cap': None,
                    'pe_ratio': None,
                    'dividend_yield': None,
                    'beta': None
                }
        
        # 保存价格数据到类属性
        self.price_data = {
            ticker: data[ticker] for ticker in self.tickers
        }
        
        # 保存基准数据到类属性并计算收益率
        self.benchmark_data = pd.DataFrame({
            'Close': data[self.benchmark]['Close'],
            'Returns': data[self.benchmark]['Close'].pct_change()  # 添加这行计算收益率
        })
        
        # 计算收益率数据
        self.returns_data = pd.DataFrame({
            ticker: self.price_data[ticker]['Close'].pct_change()
            for ticker in self.tickers
        })
        
        return data

    def calculate_technical_indicators(self, ticker_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        ticker_data: DataFrame containing historical price data, including 'Close', 'High', 'Low', 'Volume', from self.price_data
        """
        # Trend Indicators
        ticker_data['sma_20'] = ta.trend.sma_indicator(ticker_data['Close'], window=20)
        ticker_data['ema_20'] = ta.trend.ema_indicator(ticker_data['Close'], window=20)
        ticker_data['macd'] = ta.trend.macd_diff(ticker_data['Close'])
        
        # Volatility Indicators
        ticker_data['bollinger_high'] = ta.volatility.bollinger_hband(ticker_data['Close'])
        ticker_data['bollinger_low'] = ta.volatility.bollinger_lband(ticker_data['Close'])
        ticker_data['atr'] = ta.volatility.average_true_range(ticker_data['High'], 
                                                            ticker_data['Low'], 
                                                            ticker_data['Close'])
        
        # Momentum Indicators
        ticker_data['rsi'] = ta.momentum.rsi(ticker_data['Close'])
        ticker_data['stoch'] = ta.momentum.stoch(ticker_data['High'], 
                                               ticker_data['Low'], 
                                               ticker_data['Close'])
        
        return ticker_data

    def perform_risk_analysis(self) -> Dict:
        """
        Comprehensive risk analysis including various risk metrics
        """
        portfolio_returns = self.returns_data.dot(self.weights)
        benchmark_returns = self.benchmark_data['Returns']
        
        risk_metrics = {
            # Return Metrics
            'total_return': ep.cum_returns_final(portfolio_returns),
            'annual_return': ep.annual_return(portfolio_returns),
            'daily_vol': ep.annual_volatility(portfolio_returns),
            
            # Risk-adjusted Metrics
            'sharpe_ratio': ep.sharpe_ratio(portfolio_returns, risk_free=self.risk_free_rate),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns, self.risk_free_rate),  # 使用自定义函数
            'calmar_ratio': ep.calmar_ratio(portfolio_returns),
            
            # Risk Metrics
            'var_95': self._calculate_var(portfolio_returns, 0.95),
            'cvar_95': self._calculate_cvar(portfolio_returns, 0.95),
            'max_drawdown': ep.max_drawdown(portfolio_returns),
            
            # Relative Metrics
            'alpha': self._calculate_alpha(portfolio_returns, benchmark_returns),
            'beta': self._calculate_beta(portfolio_returns, benchmark_returns),
            'tracking_error': self._calculate_tracking_error(portfolio_returns, benchmark_returns),
            'information_ratio': self._calculate_information_ratio(portfolio_returns, benchmark_returns)
        }
        
        return risk_metrics

    def predict_future_returns(self, window: int = 30) -> pd.DataFrame:
        """
        Predict future returns using machine learning
        """
        # Prepare features
        feature_data = pd.DataFrame()
        for ticker in self.tickers:
            ticker_data = self.technical_indicators[ticker]
            
            # Create lagged features
            for lag in range(1, 6):
                feature_data[f'{ticker}_lag_{lag}'] = ticker_data['Close'].shift(lag)
                feature_data[f'{ticker}_vol_lag_{lag}'] = ticker_data['Volume'].shift(lag)
                
            # Add technical indicators as features
            feature_data[f'{ticker}_rsi'] = ticker_data['rsi']
            feature_data[f'{ticker}_macd'] = ticker_data['macd']
            feature_data[f'{ticker}_atr'] = ticker_data['atr']
        
        # Prepare target
        target = self.returns_data.dot(self.weights)
        
        # Split data
        X = feature_data.dropna()
        y = target.loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        return pd.DataFrame({
            'Actual': y_test,
            'Predicted': predictions
        })
    
    def create_interactive_dashboard(self) -> None:
        """
        Create interactive dashboard using Plotly
        """

        if self.returns_data is None or self.benchmark_data is None:
            raise ValueError("收益率数据未正确初始化")
    
        if 'Returns' not in self.benchmark_data.columns:
            raise ValueError("基准数据中缺少收益率数据")


        fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # 第一行：常规图和饼图
            [{"type": "polar"}, {"type": "xy"}],   # 第二行：雷达图和常规图
            [{"type": "xy"}, {"type": "xy"}],      # 第三行：两个常规图
        ],
        subplot_titles=(
            'Portfolio Performance',
            'Asset Allocation',
            'Risk Metrics',
            'Technical Indicators',
            'Return Distribution',
            'Correlation Matrix'
        )
    )
        
        # 1. Portfolio Performance Plot
        portfolio_values = (1 + self.returns_data.dot(self.weights)).cumprod()
        benchmark_values = (1 + self.benchmark_data['Returns']).cumprod()
        
        fig.add_trace(
            go.Scatter(x=portfolio_values.index, y=portfolio_values, name='Portfolio'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=benchmark_values.index, y=benchmark_values, name='Benchmark'),
            row=1, col=1
        )
        
        # 2. Asset Allocation Pie Chart
        fig.add_trace(
            go.Pie(labels=self.tickers, values=self.weights),
            row=1, col=2
        )
        
        # 3. Risk Metrics Radar Chart
        risk_metrics = self.perform_risk_analysis()
        fig.add_trace(
            go.Scatterpolar(
                r=[risk_metrics['sharpe_ratio'], risk_metrics['sortino_ratio'],
                   risk_metrics['alpha'], risk_metrics['beta'],
                   risk_metrics['information_ratio']],
                theta=['Sharpe', 'Sortino', 'Alpha', 'Beta', 'Info Ratio'],
                fill='toself'
            ),
            row=2, col=1
        )
        
        # 4. Technical Indicators
        for ticker in self.tickers:
            fig.add_trace(
                go.Scatter(
                    x=self.technical_indicators[ticker].index,
                    y=self.technical_indicators[ticker]['rsi'],
                    name=f'{ticker} RSI'
                ),
                row=2, col=2
            )
        
        # 5. Return Distribution
        portfolio_returns = self.returns_data.dot(self.weights)
        fig.add_trace(
            go.Histogram(x=portfolio_returns, nbinsx=50),
            row=3, col=1
        )
        
        # 6. Correlation Matrix
        corr_matrix = self.returns_data.corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values,
                      x=corr_matrix.index,
                      y=corr_matrix.columns),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(height=1200, showlegend=True, title_text="Portfolio Analysis Dashboard")
        
        # Save the dashboard
        fig.write_html("portfolio_dashboard.html")

    def generate_comprehensive_report(self) -> str:
        """
        Generate detailed analysis report
        """
        risk_metrics = self.perform_risk_analysis()
        predictions = self.predict_future_returns()
        
        report = f"""
        Comprehensive Portfolio Analysis Report
        =====================================
        
        1. Portfolio Overview
        -------------------
        {self._generate_portfolio_overview()}
        
        2. Performance Metrics
        -------------------
        {self._generate_performance_metrics(risk_metrics)}
        
        3. Risk Analysis
        --------------
        {self._generate_risk_analysis(risk_metrics)}
        
        4. Technical Analysis
        ------------------
        {self._generate_technical_analysis()}
        
        5. Predictive Analysis
        -------------------
        {self._generate_predictive_analysis(predictions)}
        
        6. Investment Recommendations
        -------------------------
        {self._generate_recommendations()}
        """
        
        return report


# Example usage
if __name__ == "__main__":
    try:
        # 1. 创建分析器实例
        config = {
            'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'weights': [0.2, 0.2, 0.2, 0.2, 0.2],
            'risk_free_rate': 0.02,
            'benchmark': 'SPY'
        }
        analyzer = EnhancedPortfolioAnalyzer(config)

        # 2. 设定时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        # 3. 获取数据并进行初始化处理
        print("正在获取市场数据...")
        market_data = analyzer.fetch_market_data(start_date, end_date)

        # 4. 计算技术指标
        print("正在计算技术指标...")
        analyzer.technical_indicators = {}
        for ticker in analyzer.tickers:
            analyzer.technical_indicators[ticker] = analyzer.calculate_technical_indicators(
                analyzer.price_data[ticker]
            )

        # 5. 生成可视化结果
        print("正在生成交互式仪表板...")
        analyzer.create_interactive_dashboard()

        # 6. 生成分析报告
        print("正在生成分析报告...")
        report = analyzer.generate_comprehensive_report()

        # 7. 保存报告
        with open('portfolio_analysis_report.txt', 'w') as f:
            f.write(report)

        print("\n分析完成！")
        print("- 交互式仪表板已保存为: portfolio_dashboard.html")
        print("- 分析报告已保存为: portfolio_analysis_report.txt")

    except Exception as e:
        print(f"发生错误: {str(e)}")