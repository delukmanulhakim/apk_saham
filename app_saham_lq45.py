# Import Library
import os
import sys
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu

# Local import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_update_data

warnings.filterwarnings("ignore")
def ensure_float_dataframe(df):
    return df.apply(pd.to_numeric, errors='coerce')

# Konfigurasi halaman
st.set_page_config(page_title="PREDIKSI STOCK PRICE LQ45", layout="wide")

TICKERS = [
    'ACES', 'ADMR', 'ADRO', 'AKRA', 'AMMN',
    'AMRT', 'ANTM', 'ARTO', 'ASII', 'BBCA',
    'BBNI', 'BBRI', 'BBTN', 'BMRI', 'BRIS',
    'BRPT', 'CPIN', 'CTRA', 'ESSA', 'EXCL',
    'GOTO', 'ICBP', 'INCO', 'INDF', 'INKP',
    'ISAT', 'ITMG', 'JPFA', 'JSMR', 'KLBF',
    'MAPA', 'MAPI', 'MBMA', 'MDKA', 'MEDC',
    'PGAS', 'PGEO', 'PTBA', 'SIDO', 'SMGR',
    'SMRA', 'TLKM', 'TOWR', 'UNTR', 'UNVR'
]

# Sidebar untuk konfigurasi
st.sidebar.title("KONFIGURASI DATA")
lq45_symbols = [s.replace(".csv", "") for s in os.listdir("saham_lq45_new")]
selected_symbol = st.sidebar.selectbox("Pilih kode saham:", sorted(TICKERS))
start_date = st.sidebar.date_input("Tanggal mulai", datetime(2024, 1, 2))
end_date = st.sidebar.date_input("Tanggal akhir", datetime(2025, 5,21))
model_type = st.sidebar.selectbox("Pilih Model", ["GLM", "ARIMA"])
theme_option = st.sidebar.selectbox("Pilih Tema", ["Terang", "Gelap"])

# download data terbaru
st.sidebar.title("Download Data Saham Baru")
if st.sidebar.button("Download Data Terbaru"):
    try:
        stock_data = yf.download(f"{selected_symbol}.JK", start=start_date, end=end_date)
        if not os.path.exists("saham_lq45_new"):
            os.makedirs("saham_lq45_new")
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={
            "Date": "date",
            "Open": "open_price",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)
        stock_data.to_csv(f"saham_lq45_new/{selected_symbol}.csv", index=False)
        st.sidebar.success(f"Data {selected_symbol} berhasil disimpan di folder saham_lq45_new!")
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan: {e}")

# Tema warna
if theme_option == "Gelap":
    bg_color, text_color = "#2c2c2c", "#f8f9fa"
    line_color_actual, line_color_pred = "#e63946", "#a8dadc"
else:
    bg_color, text_color = "#ffffff", "#000000"
    line_color_actual, line_color_pred = "blue", "red"


# Fungsi fitur dengan variabel independen
def create_glm_features_with_indep_vars(df, lags=5):
    df["date"] = pd.to_datetime(df["date"])
    df["weekday"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["date_ordinal"] = df["date"].map(pd.Timestamp.toordinal)

    df_feat = pd.DataFrame()

    for i in range(1, lags + 1):
        df_feat[f"close_lag_{i}"] = df["close"].shift(i)
        df_feat[f"open_lag_{i}"] = df["open_price"].shift(i)
        df_feat[f"high_lag_{i}"] = df["high"].shift(i)
        df_feat[f"low_lag_{i}"] = df["low"].shift(i)
        df_feat[f"volume_lag_{i}"] = df["volume"].shift(i)

        # fitur waktu
        df_feat[f"weekday_lag_{i}"] = df["weekday"].shift(i)
        df_feat[f"month_lag_{i}"] = df["month"].shift(i)
        df_feat[f"day_lag_{i}"] = df["day"].shift(i)
        df_feat[f"date_ordinal_lag_{i}"] = df["date_ordinal"].shift(i)

    df_feat["target"] = df["close"]
    return df_feat.dropna()

# Fungsi evaluasi model GLM
def evaluate_glm_manual(y_true, y_pred):
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, mape


#fungsi adf test
def adf_test(ts_data):
    result = adfuller(ts_data)
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "usedlag": result[2],
        "nobs": result[3],
        "critical_values": result[4],
        "icbest": result[5]
    }

# Tuning model GLM
def tune_glm_model(df_raw, max_lags=5):
    best_mape = float('inf')
    best_lags = 1
    best_model = None

    for lags in range(1, max_lags + 1):
        features = create_glm_features_with_indep_vars(df_raw, lags=lags)
        if features.empty:
            continue
        X = features.drop("target", axis=1)
        y = features["target"]
        X = X.apply(pd.to_numeric, errors='coerce').dropna()
        y = pd.to_numeric(y, errors='coerce').loc[X.index]
        X = sm.add_constant(X, has_constant='add').astype(float)
        y = y.astype(float)
        model = sm.GLM(y, X).fit()
        pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, pred) * 100  # dalam persen
        if mape < best_mape:
            best_mape = mape
            best_lags = lags
            best_model = model
    return best_model, best_lags

# Tulisan berjalan
st.markdown(
    """
    <marquee behavior="scroll" direction="left" style="color:red; font-size:20px; font-weight:bold;">
         Selamat datang di Aplikasi Prediksi Harga Saham LQ45 menggunakan Algoritma GLMs dan ARIMA
    </marquee>
    """,
    unsafe_allow_html=True
)

# menu navigasi
selected = option_menu(
    menu_title="Navigasi",
    options=["Beranda", "Konfigurasi Data", "Evaluasi Model", "Prediksi Saham"],
    icons=["house", "cog", "bar-chart", "chart-line"],
    orientation="horizontal",
)

# halaman beranda
if selected == "Beranda":
    st.title("PREDIKSI HARGA SAHAM LQ45")
    st.markdown(f"Data Historis diambil dari yahoo finance sampai tanggal 21 Mei 2025.")

# Halaman konfigurasi data
elif selected == "Konfigurasi Data":
    st.title("KONFIGURASI DATA")
    st.write(f"Kode Saham: **{selected_symbol}**")
    st.write(f"Rentang Tanggal: {start_date} hingga {end_date}")
    st.write(f"Model yang digunakan: **{model_type}**")
    try:
        file_path = f"saham_lq45_new/{selected_symbol}.csv"
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()
        df_filtered["date"] = df_filtered["date"].dt.strftime('%d-%m-%Y')
        harga_cols = ["open_price", "high", "low", "close"]
        df_filtered[harga_cols] = df_filtered[harga_cols].apply(pd.to_numeric, errors="coerce")
        df_filtered[harga_cols] = df_filtered[harga_cols].applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
        st.subheader("Data Saham yang Digunakan")
        st.dataframe(df_filtered[["date"] + harga_cols + ["volume"]])
    except FileNotFoundError:
        st.error(f"Data untuk kode **{selected_symbol}** tidak ditemukan.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")

# Halaman evaluasi
elif selected == "Evaluasi Model":
    # Load dan filter data
    df = load_and_update_data(selected_symbol)
    df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()

    # Visualisasi Harga Historis
    st.subheader("Histori Harga Saham")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["open_price"], name='Open', line=dict(color='skyblue')))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["high"], name='High', line=dict(color='limegreen')))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["low"], name='Low', line=dict(color='salmon')))
    fig.add_trace(go.Scatter(x=df_filtered["date"], y=df_filtered["close"], name='Close', line=dict(color=line_color_actual)))
    fig.update_layout(
        title="Harga Open, High, Low, Close",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        template="plotly_dark" if theme_option == "Gelap" else "plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model GLM
    if model_type == "GLM":
        st.subheader("Generalized Linear Model (GLMs)")
        best_model, best_lags = tune_glm_model(df_filtered)
        st.success(f"Model terbaik ditemukan dengan {best_lags} lag.")
        features = create_glm_features_with_indep_vars(df_filtered, lags=best_lags)
        X = features.drop(columns="target")
        y = features["target"]
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        X = sm.add_constant(X, has_constant='add').dropna()
        y = y.loc[X.index]
        X = X.astype(float)
        y = y.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = sm.GLM(y_train, X_train).fit()
        pred_test = model.predict(X_test)

        y_true = np.array(y_test, dtype=float)
        y_pred = np.array(pred_test, dtype=float)

        mse, rmse, mae, mape = evaluate_glm_manual(y_true, y_pred)

        # menampilkan nilai aktual dan prediksi
        st.markdown("### Nilai Aktual (y_true) vs Prediksi (y_pred)")
        df_hasil = pd.DataFrame({
            "y_true": y_true,
            "y_pred":np.round(y_pred, 3)
        })
        st.dataframe(df_hasil)

        # Tampilkan hasil evaluasi
        st.subheader("Hasil Evaluasi Model GLMs")
        st.write(pd.DataFrame({
            "Metrik": [
                "Mean Squared Error",
                "Root Mean Squared Error",
                "Mean Absolute Error",
                "Mean Absolute Percentage Error"
            ],
            "Nilai": [round(mse, 3), round(rmse, 3), round(mae, 3), round(mape, 3)]
        }))

        with st.expander("Ringkasan Model GLM"):
            st.code(model.summary().as_text(), language='text')

            jumlah_lag = 5
            fitur_per_lag = 9  # close, open, high, low, volume + weekday, month, day, date_ordinal
            total_fitur = jumlah_lag * fitur_per_lag

            total_data_awal = len(df)
            total_data_final = len(X_train) + len(X_test)

            # tabel ringkasan
            summary_table = pd.DataFrame({
                "Keterangan": [
                    "Jumlah Lag yang Digunakan",
                    "Jumlah Fitur per Lag",
                    "Total Fitur Hasil Lag",
                    "Total Data yang Digunakan Setelah Lag",
                    "Df Model (fitur aktif)",
                    "Df Residuals (sisa observasi)"
                ],
                "Nilai": [
                    jumlah_lag,
                    fitur_per_lag,
                    total_fitur,
                    total_data_final,
                    int(model.df_model),
                    int(model.df_resid)
                ]
            })

            st.subheader("Informasi Tambahan Model")
            st.table(summary_table)

        # Visualisasi Prediksi vs Aktual
        st.subheader("Visualisasi Prediksi vs Aktual (Per Tanggal)")
        tanggal = df_filtered["date"].iloc[-len(y_true):].dt.date
        actual_vs_pred = pd.DataFrame({
            "Tanggal": tanggal,
            "Harga Aktual": y_true,
            "Harga Prediksi": y_pred
        })
        fig1 = px.line(actual_vs_pred, x="Tanggal", y=["Harga Aktual", "Harga Prediksi"],
                       title="Grafik Harga Aktual vs Prediksi",
                       labels={"value": "Harga", "variable": "Keterangan"})
        fig1.update_layout(xaxis_title="Tanggal", yaxis_title="Harga Saham", title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

        # Korelasi
        st.subheader("Korelasi Prediksi dan Aktual")
        corr_df = pd.DataFrame({"Aktual": y_true, "Prediksi": y_pred})
        fig2 = px.imshow(corr_df.corr(), text_auto=True, title="Heatmap Korelasi")
        fig2.update_layout(title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

        # Scatter Plot
        st.subheader("Scatter Plot dan Regresi Linear")
        fig3 = px.scatter(corr_df, x="Aktual", y="Prediksi", trendline="ols",
                          title="Plot Titik dan Garis Linear Prediksi vs Aktual")
        fig3.update_layout(xaxis_title="Harga Aktual", yaxis_title="Harga Prediksi", title_x=0.5)
        st.plotly_chart(fig3, use_container_width=True)

    elif model_type == "ARIMA":
        ts_data = df_filtered.set_index("date")["close"]
        ts_data.index = pd.to_datetime(ts_data.index)
        ts_data = ts_data.astype(float).dropna()
        try:
            # Uji Stasioneritas
            st.subheader("Uji Stasioneritas Data dengan ADF Test")
            result_adf = adfuller(ts_data)
            st.write('ADF Statistic:', result_adf[0])
            st.write('p-value:', result_adf[1])
            if result_adf[1] < 0.05:
                st.success("Data stasioner (p < 0.05), tidak perlu differencing tambahan")
            else:
                st.warning("Data tidak stasioner (p >= 0.05), mungkin perlu differencing")

            split_idx = int(len(ts_data) * 0.9)
            train, test = ts_data.iloc[:split_idx], ts_data.iloc[split_idx:]
            # parameter
            model = ARIMA(train, order=(1, 1, 2))
            model_fit = model.fit()
            st.info("Model ARIMA terlatih dengan order (1,1,2)")

            # Prediksi
            forecast = model_fit.forecast(steps=len(test))
            y_true = test.values
            y_pred = forecast.values
            tanggal = test.index
            df_compare = pd.DataFrame({
                "y_true": y_true,
                "y_pred": np.round(y_pred, 3)
            })

            st.dataframe(df_compare)

            # Evaluasi
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            # Tampilkan hasil evaluasi
            st.subheader("Hasil Evaluasi Model ARIMA")
            st.write(pd.DataFrame({
                "Metrik": ["Mean Absolute Percentage Error"],
                "Nilai": [mape]

            }))

            # Plot residual
            st.subheader("Plot Residual Model ARIMA")
            residuals = model_fit.resid
            fig2, axs = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(residuals, bins=30, kde=True, ax=axs[0])
            axs[0].set_title('Distribusi Residual')
            sns.lineplot(x=range(len(residuals)), y=residuals, ax=axs[1])
            axs[1].set_title('Plot Residual over Time')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Residual')
            plt.tight_layout()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Gagal menjalankan evaluasi ARIMA: {e}")

# Halaman prediksi
elif selected == "Prediksi Saham":
    st.title("PREDIKSI HARGA SAHAM")
    df = load_and_update_data(selected_symbol)
    df_filtered = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()
    if df_filtered.empty:
        st.error("Data kosong untuk rentang tanggal tersebut.")
        st.stop()
    close_prices = df_filtered["close"].dropna().reset_index(drop=True)
    def plot_forecast(actual_series, forecast_value):
        plt.figure(figsize=(10, 5))
        plt.plot(actual_series.index, actual_series.values, label="Data Aktual")
        plt.scatter(actual_series.index[-1] + 1, forecast_value, color='red', label="Prediksi Berikutnya")
        plt.title("Harga Saham: Aktual dan Prediksi Berikutnya")
        plt.xlabel("Waktu")
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

    if model_type == "GLM":
        st.subheader("Generalized Linear Model (GLMs)")
        best_model, best_lags = tune_glm_model(df_filtered)
        features = create_glm_features_with_indep_vars(df_filtered, lags=best_lags)
        features = features.dropna()
        if not features.empty:
            X = features.drop("target", axis=1)
            y = features["target"]
            X = sm.add_constant(X, has_constant='add')
            X = X.astype(float)
            pred = best_model.predict(X)
            n_days = st.number_input("Jumlah hari prediksi:", 1, 30, 3)
            future_df = df_filtered.copy()
            future_preds = []

            for _ in range(n_days):
                df_feat = create_glm_features_with_indep_vars(future_df, lags=best_lags)
                if df_feat.empty or "target" not in df_feat.columns:
                    st.warning("Fitur kosong atau target tidak ditemukan.")
                    break
                last_input = df_feat.iloc[-1:].drop(columns="target")
                last_input = sm.add_constant(last_input, has_constant='add').astype(float)
                try:
                    pred_series = best_model.predict(last_input)
                    pred_price = pred_series.iloc[0]
                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")
                    break
                future_preds.append(pred_price)
                next_date = future_df["date"].iloc[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame([{
                    "date": next_date,
                    "open_price": pred_price,
                    "high": pred_price,
                    "low": pred_price,
                    "close": pred_price,
                    "volume": 0
                }])
                future_df = pd.concat([future_df, new_row], ignore_index=True)

            if future_preds:
                st.subheader("HASIL PREDIKSI")
                future_dates = pd.date_range(start=future_df["date"].iloc[-n_days], periods=n_days)
                result_df = pd.DataFrame({
                    "Tanggal": future_dates.date,
                    "Harga Prediksi": future_preds
                })
                st.dataframe(result_df)
                fig = px.line(result_df, x="Tanggal", y="Harga Prediksi", title="Grafik Harga Saham Hasil Prediksi")
                fig.update_layout(xaxis_title="Tanggal", yaxis_title="Harga", xaxis_tickformat="%Y-%m-%d")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada hasil prediksi yang dihasilkan.")

    elif model_type == "ARIMA":
        ts_data = df_filtered.set_index("date")["close"]
        ts_data.index = pd.to_datetime(ts_data.index)
        ts_data = ts_data.astype(float).dropna()
        try:
            model = ARIMA(ts_data, order=(1, 1, 2))
            model_fit = model.fit()
            # menginput jumlah hari ke-n untuk prediksi ke depan
            st.subheader("ARIMA")
            n_days = st.number_input("Jumlah hari prediksi ke depan:", min_value=1, max_value=30, value=1)

            # Prediksi masa depan
            future_preds = model_fit.forecast(steps=n_days)
            last_date = ts_data.index[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 1)]
            df_forecast = pd.DataFrame({
                "Tanggal": future_dates,
                "Prediksi_ARIMA": future_preds
            }).set_index("Tanggal")

            # Tampilkan hasil prediksi
            st.write("Hasil Prediksi ARIMA:")
            st.dataframe(df_forecast)

            # Grafik prediksi
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts_data.index, ts_data, label="Data Aktual")
            ax.plot(df_forecast.index, df_forecast["Prediksi_ARIMA"], label="Prediksi ke Depan", linestyle='--',
                    color="orange")
            ax.set_title("Prediksi Harga Saham Menggunakan ARIMA")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga Saham")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal menjalankan prediksi ARIMA: {e}")

