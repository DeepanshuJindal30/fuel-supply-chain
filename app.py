import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Define the fuel supply chain model
class FuelSupplyChain:
    def __init__(self, demand, supply, transport_capacity, lead_time):
        # Ensure demand and transport capacity are always treated as arrays
        self.demand = np.array(demand) if not isinstance(demand, np.ndarray) else demand
        self.supply = supply
        self.transport_capacity = transport_capacity  # Transport capacity is treated as a scalar
        self.lead_time = lead_time

    def forecast_demand(self, months=12):
        # Prophet for more sophisticated demand forecasting
        df = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=len(self.demand), freq='M'),
            'y': self.demand
        })
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=months, freq='M')
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(months)

    def optimize_supply_chain(self):
        # Ensure transport capacity is an array of size 12 (to match demand)
        transport_capacity_array = np.full_like(self.demand, self.transport_capacity)  # Create an array of transport capacity

        # Normalize demand and transport capacity using MinMaxScaler
        demand_scaled = MinMaxScaler().fit_transform(self.demand.reshape(-1, 1))
        transport_scaled = MinMaxScaler().fit_transform(transport_capacity_array.reshape(-1, 1))
        
        # Prepare the feature matrix X and target vector y for regression
        X = np.concatenate([demand_scaled, transport_scaled], axis=1)
        y = self.supply * np.ones_like(self.demand)  # Target is the supply value
        
        # Fit a Linear Regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict optimized supply based on the trained model
        optimized_supply = model.predict(X)
        
        return optimized_supply

    def generate_candlestick_data(self, forecast):
        # Generate candlestick data: open, high, low, close
        open_supply = self.supply + np.random.normal(0, 50, 12)
        high_supply = np.minimum(self.supply + self.transport_capacity, forecast['yhat'] * 1.2 + np.random.normal(0, 50, 12))
        low_supply = np.maximum(self.supply - self.transport_capacity, forecast['yhat_lower'] + np.random.normal(0, 50, 12))
        close_supply = np.minimum(self.supply, forecast['yhat'] + np.random.normal(0, 50, 12))

        return pd.DataFrame({
            'Month': np.arange(1, 13),
            'Open': open_supply,
            'High': high_supply,
            'Low': low_supply,
            'Close': close_supply
        })

    def visualize_supply_chain_candles(self, forecast):
        df = self.generate_candlestick_data(forecast)

        fig = go.Figure(data=[go.Candlestick(
            x=df['Month'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Fuel Supply'
        )])

        df['SMA_3'] = df['Close'].rolling(window=3).mean()
        
        fig.add_trace(go.Scatter(
            x=df['Month'],
            y=df['SMA_3'],
            mode='lines',
            name='3-Month Moving Average',
            line=dict(color='blue', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Fuel Supply Chain Forecast (Candlestick)',
            xaxis_title='Month',
            yaxis_title='Fuel Supply (Units)',
            template='plotly_dark',
            xaxis=dict(tickmode='linear'),
            yaxis=dict(range=[df['Low'].min() - 50, df['High'].max() + 50])
        )

        st.plotly_chart(fig)

# Streamlit App UI
def main():
    st.title("AI-Powered Fuel Supply Chain Optimization")

    st.sidebar.header("Configure Supply Chain Parameters")

    demand_input = st.sidebar.slider('Fuel Demand (units)', min_value=100, max_value=1000, value=500)
    supply_input = st.sidebar.slider('Fuel Supply (units)', min_value=100, max_value=1000, value=400)
    transport_capacity_input = st.sidebar.slider('Transport Capacity (units)', min_value=50, max_value=500, value=300)
    lead_time_input = st.sidebar.slider('Lead Time (months)', min_value=1, max_value=6, value=3)

    # Create supply chain model with an array for demand
    demand_array = np.full(12, demand_input)
    supply_chain = FuelSupplyChain(demand=demand_array, supply=supply_input,
                                   transport_capacity=transport_capacity_input, lead_time=lead_time_input)

    forecast = supply_chain.forecast_demand()

    supply_chain.visualize_supply_chain_candles(forecast)

    # Show a summary of the model
    st.subheader("Supply Chain Parameters")
    st.write(f"Demand: {demand_input} units")
    st.write(f"Supply: {supply_input} units")
    st.write(f"Transport Capacity: {transport_capacity_input} units")
    st.write(f"Lead Time: {lead_time_input} months")

    # AI Optimization Insights
    optimized_supply = supply_chain.optimize_supply_chain()
    st.subheader("AI Optimization Insights")
    st.write(f"Optimized Supply (after AI adjustments): {optimized_supply[-1]} units")  # Show the latest optimized supply

if __name__ == "__main__":
    main()
    st.markdown("<br><hr><p style='text-align: center;'>Made with ♥ by Anshika Jadon</p>", unsafe_allow_html=True)

