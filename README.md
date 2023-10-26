# Trading-Bot-Project
Coding Experiments
# Forex Market Analysis Project

## Description

This project involves the analysis of historical forex market data, focusing on the EUR/USD currency pair. The aim is to calculate a variety of technical analysis indicators to explore possible trading strategies and predictions. The code provided processes historical candlestick data and computes several technical indicators for market analysis. 

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn

These libraries are required for data handling, analysis, and machine learning functions. Make sure you have them installed.

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The dataset `EURUSD_Candlestick_1_h_ASK_01.01.2019-28.03.2020.csv` must be placed in the `DATA` folder. This CSV file should contain historical data for the EUR/USD currency pair, including columns for Date, Open, High, Low, Close, and Volume.

## Usage

To use this code, you need to include your functions for the following indicators in `Experimentation/functions`:
- Momentum
- Stochastic
- Williams %R
- Price Rate of Change (PROC)
- Williams Accumulation Distribution Line (WADL)
- Average Directional Index Oscillator (ADOS)
- Commodity Channel Index (CCI)
- Bollinger Bands
- Heikin Ashi
- Period Averages
- Slopes

The script performs the following operations:
1. Loads the dataset and renames columns for uniformity.
2. Calculates various financial indicators using different time periods as specified in the script.
3. Creates a master DataFrame `masterFrame` containing all the calculated features.
4. Processes the master DataFrame to handle missing values and calculate percentage changes in closing prices.
5. Splits the cleaned data into training and testing datasets (this part is commented and needs to be adapted based on your ML model and strategy).

Make sure to modify and uncomment the `train_test_split` section at the end of the script according to your machine learning setup.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](#) if you want to contribute.

## License

State your licensing terms or simply link to the LICENSE file.

---

**Note:** This README assumes that the user has a basic understanding of technical analysis in forex markets and Python programming. Adjustments may be necessary depending on the detailed functionality of the functions imported from `Experimentation/functions`.
