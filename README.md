# investment-portfolio

# Enhanced Portfolio Analyzer

A comprehensive Python tool for portfolio analysis, technical indicators calculation, risk assessment, and investment recommendations.

## Features

- **Market Data Fetching**: Automated retrieval of historical price data and fundamental metrics using yfinance
- **Technical Analysis**: Calculation of various technical indicators including:
  - Moving Averages (SMA, EMA)
  - MACD
  - RSI
  - Bollinger Bands
  - ATR (Average True Range)
  - Stochastic Oscillator
- **Risk Analysis**: Comprehensive risk metrics calculation:
  - Value at Risk (VaR)
  - Conditional VaR (CVaR)
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Alpha and Beta
  - Information Ratio
  - Tracking Error
- **Machine Learning Integration**: Future returns prediction using Random Forest
- **Interactive Visualization**: Dynamic dashboards using Plotly with:
  - Portfolio Performance Charts
  - Asset Allocation Views
  - Risk Metrics Radar Charts
  - Technical Indicators Visualization
  - Return Distribution Analysis
  - Correlation Matrix Heatmaps
- **Automated Reporting**: Generation of detailed analysis reports with investment recommendations

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- pandas
- numpy
- yfinance
- ta
- scikit-learn
- plotly
- scipy
- empyrical

## Usage

1. Create a configuration dictionary with your portfolio details:

```python
config = {
    'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
    'weights': [0.2, 0.2, 0.2, 0.2, 0.2],
    'risk_free_rate': 0.02,
    'benchmark': 'SPY'
}
```

2. Initialize the analyzer and run analysis:

```python
from datetime import datetime, timedelta
from portfolio_analyzer import EnhancedPortfolioAnalyzer

# Create analyzer instance
analyzer = EnhancedPortfolioAnalyzer(config)

# Set time range
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Fetch data and perform analysis
market_data = analyzer.fetch_market_data(start_date, end_date)

# Calculate technical indicators
analyzer.technical_indicators = {}
for ticker in analyzer.tickers:
    analyzer.technical_indicators[ticker] = analyzer.calculate_technical_indicators(
        analyzer.price_data[ticker]
    )

# Generate visualizations and reports
analyzer.create_interactive_dashboard()
report = analyzer.generate_comprehensive_report()
```

## Project Structure

- `main.py`: Main implementation of the EnhancedPortfolioAnalyzer class
- `helper_compute_index.py`: Base class with core computation methods
- `helper_generate_report.py`: Report generation functionality
- Generated outputs:
  - `portfolio_dashboard.html`: Interactive visualization dashboard
  - `portfolio_analysis_report.txt`: Detailed analysis report

## Output Examples

The analyzer generates two main outputs:

1. An interactive dashboard (`portfolio_dashboard.html`) with:
   - Portfolio performance comparison
   - Asset allocation visualization
   - Risk metrics overview
   - Technical indicators charts
   - Return distribution analysis
   - Correlation matrix

2. A comprehensive report (`portfolio_analysis_report.txt`) containing:
   - Portfolio overview
   - Performance metrics
   - Risk analysis
   - Technical analysis
   - Predictive analysis
   - Investment recommendations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- `yfinance` for market data retrieval
- `ta` library for technical analysis calculations
- `plotly` for interactive visualizations
- `scikit-learn` for machine learning capabilities
- `empyrical` for financial calculations
