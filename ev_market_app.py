import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import io  # ✅ For memory-safe CSV download

# Page settings
st.set_page_config(page_title="EV Dashboard", layout="wide")
sns.set_style("whitegrid")

# Optional CSS
def local_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
local_css()

st.title("🚗 Electric Vehicle Market Dashboard")
st.markdown("Get insights into EV trends, top brands, and future predictions. Upload your own dataset or use the default.")

# Load data
def load_data():
    return pd.read_csv('Electric_Vehicle_Population_Data.csv')

uploaded_file = st.sidebar.file_uploader("Upload your EV CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

# Data cleaning
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
df = df.dropna(subset=['Model Year', 'Electric Vehicle Type', 'Make', 'Electric Range'])
df['Model Year'] = df['Model Year'].astype(int)

# Sidebar filters
min_year, max_year = df['Model Year'].min(), df['Model Year'].max()
year_range = st.sidebar.slider("Filter by Model Year", int(min_year), int(max_year), (int(min_year), int(max_year)))
df_filtered = df[(df['Model Year'] >= year_range[0]) & (df['Model Year'] <= year_range[1])]

selected_make = st.sidebar.selectbox("Filter by Brand", ['All'] + sorted(df_filtered['Make'].unique()))
if selected_make != 'All':
    df_filtered = df_filtered[df_filtered['Make'] == selected_make]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total EVs", f"{len(df_filtered):,}")
col2.metric("Unique Brands", df_filtered['Make'].nunique())
col3.metric("Avg Range", f"{df_filtered['Electric Range'].mean():.1f} mi")

# ✅ Download button with memory-safe StringIO
csv_buffer = io.StringIO()
df_filtered.to_csv(csv_buffer, index=False)
st.download_button(
    label="⬇️ Download Filtered Data",
    data=csv_buffer.getvalue(),
    file_name="ev_filtered.csv",
    mime="text/csv"
)

# Tabs layout
tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔮 Forecast", "📊 Distribution"])

# --- Trends Tab ---
with tab1:
    st.subheader("📈 EV Registrations Over Time")
    yearly_counts = df_filtered.groupby('Model Year').size()
    if not yearly_counts.empty:
        fig1, ax1 = plt.subplots()
        sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o', ax=ax1)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of EVs")
        st.pyplot(fig1)
    else:
        st.warning("No data available for selected filters.")

# --- Forecast Tab ---
with tab2:
    st.subheader("🔮 Forecasting Future EV Growth")
    if len(yearly_counts) > 1:
        X = np.array(yearly_counts.index).reshape(-1, 1)
        y = yearly_counts.values
        model = LinearRegression()
        model.fit(X, y)
        future_years = np.arange(max(yearly_counts.index)+1, max(yearly_counts.index)+6).reshape(-1, 1)
        predictions = model.predict(future_years)

        fig2, ax2 = plt.subplots()
        ax2.plot(yearly_counts.index, y, label='Actual')
        ax2.plot(future_years.flatten(), predictions, label='Forecast', linestyle='--')
        ax2.set_title('EV Forecast (Next 5 Years)')
        ax2.set_xlabel("Year")
        ax2.set_ylabel("EV Registrations")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Not enough data to generate forecast.")

# --- Distribution Tab ---
with tab3:
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("🏆 Top EV Brands")
        top_makes = df_filtered['Make'].value_counts().head(10)
        st.bar_chart(top_makes)

    with col5:
        st.subheader("🚙 EV Type Distribution")
        ev_types = df_filtered['Electric Vehicle Type'].value_counts()
        fig3, ax3 = plt.subplots()
        ax3.pie(ev_types, labels=ev_types.index, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')
        st.pyplot(fig3)

    st.subheader("🔋 Electric Range Distribution")
    fig4, ax4 = plt.subplots()
    sns.histplot(df_filtered['Electric Range'], bins=30, kde=True, ax=ax4)
    ax4.set_xlabel("Range (miles)")
    st.pyplot(fig4)
