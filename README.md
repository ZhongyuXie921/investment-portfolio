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

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio-analysis-dashboard.git
cd portfolio-analysis-dashboard
```

2. Install required packages:
```bash
pip install -r requirements.txt
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
![Dashboard Preview](portfolio_dashboard.png)
*Interactive dashboard showing portfolio performance, risk metrics, and technical indicators*

### Analysis Report Sample
```
Asset Allocation:
- AAPL: 20.0%
- GOOGL: 20.0%
- MSFT: 20.0%
- AMZN: 20.0%
- META: 20.0%

Performance Metrics:
- Total Return: 36.64%
- Annual Return: 36.81%
- Daily Volatility: 20.23%
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
