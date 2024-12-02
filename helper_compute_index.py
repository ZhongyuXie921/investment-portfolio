import pandas as pd
import numpy as np
import empyrical as ep
from typing import List, Dict, Union, Optional

class BasePortfolioAnalyzer:
    def __init__(self):
        self.returns_data = None
        self.tickers = None
        self.weights = None

    # Computation methods
    def _calculate_var(self, returns: pd.Series, alpha: float) -> float:
        """
        计算 Value at Risk (VaR)
        """
        return np.percentile(returns, 100*(1-alpha))
        
    def _calculate_cvar(self, returns, confidence):
        # 计算条件风险价值
        var = self._calculate_var(returns, confidence)
        return -returns[returns <= -var].mean()

    def _calculate_alpha(self, portfolio_returns, benchmark_returns):
        # 计算阿尔法
        return ep.alpha(portfolio_returns, benchmark_returns)

    def _calculate_beta(self, portfolio_returns, benchmark_returns):
        # 计算贝塔
        return ep.beta(portfolio_returns, benchmark_returns)

    def _calculate_tracking_error(self, portfolio_returns, benchmark_returns):
        # 计算跟踪误差
        return np.std(portfolio_returns - benchmark_returns) * np.sqrt(252)

    def _calculate_information_ratio(self, portfolio_returns, benchmark_returns):
        # 计算信息比率
        tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
        return (portfolio_returns.mean() - benchmark_returns.mean()) * 252 / tracking_error

    def _calculate_prediction_confidence(self, predictions):
        # 计算预测置信度
        mse = ((predictions['Actual'] - predictions['Predicted'])**2).mean()
        return 1 - np.sqrt(mse)
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        计算 Sortino Ratio
        只考虑下行波动率的夏普比率变体
        """
        # 计算超额收益
        excess_returns = returns - risk_free_rate
            
        # 计算年化收益率
        expected_return = np.mean(excess_returns) * 252
            
        # 计算下行波动率
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252)
            
        # 如果没有下行波动，返回 NaN
        if downside_std == 0:
            return np.nan
            
        return expected_return / downside_std
        
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """
        计算投资组合的行业暴露度
        """
        # 定义每个股票的行业
        sector_mapping = {
                'AAPL': 'Technology',
                'GOOGL': 'Technology',
                'MSFT': 'Technology',
                'AMZN': 'Consumer Cyclical',
                'META': 'Technology'
            }
            
            # 计算每个行业的权重
        sector_exposure = {}
        for ticker, weight in zip(self.tickers, self.weights):
            sector = sector_mapping.get(ticker, 'Other')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
            
        return sector_exposure

    def _calculate_portfolio_volatility(self) -> float:
        """
        计算投资组合的年化波动率
        """
        portfolio_returns = self.returns_data.dot(self.weights)
        return np.std(portfolio_returns) * np.sqrt(252)

    def _calculate_average_correlation(self) -> float:
        """
        计算投资组合中股票的平均相关系数
        """
        # 计算相关系数矩阵
        corr_matrix = self.returns_data.corr()
            
        # 获取上三角矩阵的值（不包括对角线）
        upper_triangle = np.triu(corr_matrix, k=1)
            
        # 计算平均相关系数
        # 注意：需要排除0值（对角线和下三角的值）
        correlations = upper_triangle[upper_triangle != 0]
        return np.mean(correlations)