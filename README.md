# NYC Taxi Demand Forecasting

This project forecasts hourly taxi demand in New York City using deep learning models implemented in PyTorch. The notebook demonstrates how to prepare time series data and apply sequence modeling techniques to predict future demand.

## 🚕 Dataset

- **Source:** `nyc_taxi.csv`
- **Feature:** `value` (number of taxi rides)
- **Timestamp:** Hourly observations across several months

## 🔧 Preprocessing

- Convert timestamp to datetime
- Normalize `value` column using `MinMaxScaler`
- Prepare sliding windows for sequence prediction

## 🧠 Model Architecture

- Sequence prediction model using:
  - Fully connected (FC) layers or
  - Recurrent layers like LSTM (if used)
- Built with **PyTorch**
- Optimized using Adam
- Loss: MSELoss

## 📊 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## 📈 Visualization

- Actual vs. Predicted demand plot
- Residual analysis
- Loss curve

## 🧰 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchinfo
