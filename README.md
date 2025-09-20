# NIFTY 500 Momentum Stock Screener

A Streamlit web application for screening NIFTY 500 stocks based on technical analysis indicators.

## Features

- **Technical Screening**: Filters stocks based on ADX, Moving Averages, OBV, and Directional Movement
- **Risk/Reward Analysis**: Calculates risk-reward ratios using ATR-based stop losses and targets  
- **Interactive Charts**: View technical charts with moving averages and volume indicators
- **Customizable Parameters**: Adjust screening criteria via sidebar controls
- **Export Results**: Download screening results as CSV files

## Screening Criteria

The screener identifies stocks with:
- ADX > 25 (Strong trend)
- 20 DMA > 50 DMA (Bullish trend)
- Close < 20 DMA (Pullback entry opportunity)
- OBV > OBV MA20 (Volume confirmation)
- +DI > -DI (Bullish momentum)
- Risk/Reward Ratio ≥ 1.5

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nifty500-screener.git
cd nifty500-screener
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

## Deployment

This app is designed to be deployed on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

## Technical Indicators Used

- **ADX (Average Directional Index)**: Measures trend strength
- **+DI/-DI**: Directional Movement indicators for trend direction
- **SMA (Simple Moving Averages)**: 20-day and 50-day trends
- **ATR (Average True Range)**: Volatility measurement for risk management
- **OBV (On-Balance Volume)**: Volume-price trend indicator

## Usage

1. Adjust screening parameters using the sidebar
2. Click "Run Screening" to analyze stocks
3. Review results in the interactive table
4. Select stocks to view their technical charts
5. Export results for further analysis

## Disclaimer

This tool is for educational and informational purposes only. Not financial advice. Always do your own research before making investment decisions.

## License

MIT License
