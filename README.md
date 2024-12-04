# investment-portfolio

# Enhanced Portfolio Analysis Dashboard

An advanced financial analysis tool that provides comprehensive portfolio analysis, including technical indicators, risk metrics, predictive analytics, and interactive visualizations.

## Features

### 1. Comprehensive Data Analysis
- Real-time market data fetching using Yahoo Finance API
- Technical indicator calculations (RSI, MACD, Bollinger Bands, etc.)
- Advanced risk metrics computation
- Machine learning-based return predictions

### 2. Risk Analysis
- Value at Risk (VaR) calculations
- Conditional Value at Risk (CVaR)
- Maximum drawdown analysis
- Alpha and Beta calculations
- Sharpe and Sortino ratios
- Tracking error and information ratio

### 3. Technical Analysis
- Multiple technical indicators including:
  - Moving Averages (SMA, EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - ATR (Average True Range)

### 4. Interactive Visualizations
The dashboard includes:
- Portfolio performance comparison
- Asset allocation visualization
- Risk metrics radar chart
- Technical indicators tracking
- Return distribution analysis
- Correlation matrix heatmap

### 5. Automated Reporting
- Comprehensive PDF reports
- Investment recommendations
- Portfolio adjustment suggestions
- Sector exposure analysis

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-analysis-dashboard.git
cd portfolio-analysis-dashboard
```

## Project Structure
```
portfolio-analysis-dashboard/
├── main.py                         # Main analysis script
├── helper_compute_index.py         # Base computation methods
├── helper_generate_report.py       # Report generation utilities
├── portfolio_dashboard.html        # Interactive dashboard output
├── portfolio_analysis_report.txt   # Detailed analysis report
├── requirements.txt               
└── README.md
```

## Usage

1. Configure your portfolio in the main script:
```python
config = {
    'tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
    'weights': [0.2, 0.2, 0.2, 0.2, 0.2],
    'risk_free_rate': 0.02,
    'benchmark': 'SPY'
}
```

2. Run the analysis:
```bash
python main.py
```

## Output Examples

The tool generates various interactive visualizations and reports:

### Interactive Dashboard
The tool generates an interactive HTML dashboard (`portfolio_dashboard.html`) that includes:
1. Portfolio Performance Chart
2. Asset Allocation Pie Chart
3. Risk Metrics Radar Chart
4. Technical Indicators Time Series
5. Return Distribution Histogram
6. Correlation Matrix Heatmap

Users can interact with the dashboard to:
- Zoom in/out on charts
- Hover over data points for detailed information
- Toggle different metrics on/off
- Download the charts as PNG files

### Sample Analysis Report
The tool generates a comprehensive analysis report (`portfolio_analysis_report.txt`) that includes:

```
Comprehensive Portfolio Analysis Report
=====================================

1. Portfolio Overview
-------------------
Asset Allocation:
- AAPL: 20.0%
- GOOGL: 20.0%
- MSFT: 20.0%
- AMZN: 20.0%
- META: 20.0%

2. Performance Metrics
-------------------
Total Return: 36.64%
Annual Return: 36.81%
Daily Volatility: 20.23%
Sharpe Ratio: -23.25
Information Ratio: 0.31

3. Technical Analysis
------------------
AAPL:
- RSI: 63.19 (Neutral)
- MACD: 1.03 (Bullish)
MSFT:
- RSI: 53.25 (Neutral)
- MACD: 0.69 (Bullish)
[...]
```

## Dependencies
- pandas
- numpy
- yfinance
- plotly
- scikit-learn
- ta (Technical Analysis library)
- empyrical
- scipy

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
For any questions or suggestions, please open an issue in the repository.

---
**Note**: This tool is for educational and research purposes only. Always conduct thorough research and consult with financial professionals before making investment decisions.
