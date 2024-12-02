# Helper methods for report generation
import pandas as pd
from typing import List, Dict, Union, Optional

class ReportGeneratorBase:
    def __init__(self):
        self.tickers = None
        self.weights = None
        self.benchmark = None
        self.returns_data = None
        self.technical_indicators = None

    def _generate_portfolio_overview(self) -> str:
        return "\n".join([
            f"Asset Allocation:",
            *[f"- {ticker}: {weight*100:.1f}%" for ticker, weight in zip(self.tickers, self.weights)],
            f"\nBenchmark: {self.benchmark}",
            f"Analysis Period: {self.returns_data.index[0].strftime('%Y-%m-%d')} to {self.returns_data.index[-1].strftime('%Y-%m-%d')}"
        ])

    def _generate_performance_metrics(self, metrics: Dict) -> str:
        return f"""
        Total Return: {metrics['total_return']*100:.2f}%
        Annual Return: {metrics['annual_return']*100:.2f}%
        Daily Volatility: {metrics['daily_vol']*100:.2f}%
        Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
        Information Ratio: {metrics['information_ratio']:.2f}
        """

    def _generate_risk_analysis(self, metrics: Dict) -> str:
        return f"""
        Value at Risk (95%): {metrics['var_95']*100:.2f}%
        Conditional VaR (95%): {metrics['cvar_95']*100:.2f}%
        Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%
        Beta: {metrics['beta']:.2f}
        Alpha: {metrics['alpha']*100:.2f}%
        """

    def _generate_technical_analysis(self) -> str:
        analysis = []
        for ticker in self.tickers:
            data = self.technical_indicators[ticker]
            current_rsi = data['rsi'].iloc[-1]
            current_macd = data['macd'].iloc[-1]
            
            analysis.append(f"\n{ticker}:")
            analysis.append(f"- RSI: {current_rsi:.2f} ({'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'})")
            analysis.append(f"- MACD: {current_macd:.2f} ({'Bullish' if current_macd > 0 else 'Bearish'})")
            
        return "\n".join(analysis)

    def _generate_predictive_analysis(self, predictions: pd.DataFrame) -> str:
        return f"""
        Predicted Return (Next Period): {predictions['Predicted'].iloc[-1]*100:.2f}%
        Prediction Confidence: {self._calculate_prediction_confidence(predictions)*100:.2f}%
        """

    def _generate_recommendations(self) -> str:
        return "\n".join([
            "Based on the analysis:",
            *[f"- {ticker}: {self._get_recommendation(ticker)}" for ticker in self.tickers],
            "\nPortfolio Adjustment Suggestions:",
            *self._get_portfolio_suggestions()
        ])

    def _get_recommendation(self, ticker: str) -> str:
        """
        基于技术指标和风险指标为单个股票生成建议
        """
        # 获取该股票的技术指标数据
        data = self.technical_indicators[ticker]
        current_rsi = data['rsi'].iloc[-1]
        current_macd = data['macd'].iloc[-1]
        
        # 基于技术指标生成建议
        if current_rsi > 70:
            return "Consider taking profit - Overbought conditions"
        elif current_rsi < 30:
            return "Consider buying - Oversold conditions"
        elif current_macd > 0:
            return "Hold with bullish bias"
        elif current_macd < 0:
            return "Hold with cautious stance"
        else:
            return "Hold current position"

    def _get_portfolio_suggestions(self) -> List[str]:
        """
        基于整体组合分析生成调整建议
        """
        suggestions = []
        
        # 检查行业集中度
        sector_exposure = self._calculate_sector_exposure()
        if sector_exposure['Technology'] > 0.5:  # 如果科技股超过50%
            suggestions.append("- Consider reducing technology sector exposure")
        
        # 检查波动率
        portfolio_volatility = self._calculate_portfolio_volatility()
        if portfolio_volatility > 0.2:  # 如果年化波动率超过20%
            suggestions.append("- Consider adding defensive stocks to reduce volatility")
        
        # 检查相关性
        avg_correlation = self._calculate_average_correlation()
        if avg_correlation > 0.7:  # 如果平均相关性过高
            suggestions.append("- Look for uncorrelated assets to improve diversification")
        
        # 如果没有发现问题，给出维持建议
        if not suggestions:
            suggestions.append("- Current portfolio allocation appears optimal")
        
        return suggestions