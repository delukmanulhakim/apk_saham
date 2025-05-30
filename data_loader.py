import os
import pandas as pd
import streamlit as st
from datetime import datetime
import time
import random
from alpha_vantage.timeseries import TimeSeries

# API Key Alpha Vantage
API_KEY = 'GMWU4N0PYGZQQ91O'

# Daftar ticker 45 saham LQ45
TICKERS = [
    'ACES.JK', 'ADMR.JK', 'ADRO.JK', 'AKRA.JK', 'AMMN.JK',
    'AMRT.JK', 'ANTM.JK', 'ARTO.JK', 'ASII.JK', 'BBCA.JK',
    'BBNI.JK', 'BBRI.JK', 'BBTN.JK', 'BMRI.JK', 'BRIS.JK',
    'BRPT.JK', 'CPIN.JK', 'CTRA.JK', 'ESSA.JK', 'EXCL.JK',
    'GOTO.JK', 'ICBP.JK', 'INCO.JK', 'INDF.JK', 'INKP.JK',
    'ISAT.JK', 'ITMG.JK', 'JPFA.JK', 'JSMR.JK', 'KLBF.JK',
    'MAPA.JK', 'MAPI.JK', 'MDMA.JK', 'MDKA.JK', 'MEDC.JK',
    'PGAS.JK', 'PGEO.JK', 'PTBA.JK', 'SIDO.JK', 'SMGR.JK',
    'SMRA.JK', 'TLKM.JK', 'TOWR.JK', 'UNTR.JK', 'UNVR.JK'
]

# Folder penyimpanan file CSV
FOLDER_OUTPUT = 'saham_lq45_new'
os.makedirs(FOLDER_OUTPUT, exist_ok=True)

# Konstanta
MAX_RETRIES = 5  # Jumlah maksimal percobaan saat mengunduh data yang gagal


def download_with_retry(ticker):
    """Download data saham dengan retry jika terjadi error atau rate limit."""
    ts = TimeSeries(key=API_KEY, output_format='pandas')

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Mengambil data saham dengan interval daily dari Alpha Vantage
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            return data
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise e
            wait_time = random.uniform(5, 10)  # Delay acak antar retry
            print(f"[{ticker}] Terjadi kesalahan. Coba lagi dalam {wait_time:.1f}s...")
            time.sleep(wait_time)


def update_data_saham():
    """Mengunduh ulang seluruh data saham LQ45 dari Alpha Vantage."""
    st.subheader("🔄 Proses Update Data Saham LQ45")

    progress_bar = st.progress(0)
    status_text = st.empty()

    failed_tickers = []
    success_tickers = []

    total = len(TICKERS)
    batch_size = 10  # Ukuran batch (10 saham per batch)
    batches = [TICKERS[i:i + batch_size] for i in range(0, len(TICKERS), batch_size)]

    for batch_idx, batch in enumerate(batches):
        st.write(f"🔄 Memproses Batch {batch_idx + 1}/{len(batches)}...")

        for idx, ticker in enumerate(batch):
            try:
                df = download_with_retry(ticker)

                if not df.empty:
                    df = df[['4. close']]  # Mengambil hanya kolom harga penutupan
                    df.columns = ['close']  # Mengganti nama kolom
                    df.index.name = 'date'
                    filepath = os.path.join(FOLDER_OUTPUT, f"{ticker}.csv")
                    df.to_csv(filepath)
                    success_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)

            except Exception as e:
                failed_tickers.append(ticker)

            # Delay antar ticker untuk menghindari rate limit
            time.sleep(random.uniform(1.5, 4.0))  # Delay acak antar ticker

            # Update progress
            progress = ((batch_idx * batch_size) + (idx + 1)) / total
            progress_bar.progress(progress)
            status_text.text(f"Memproses {ticker} ({(batch_idx * batch_size) + (idx + 1)}/{total})...")

        # Delay antar batch untuk menghindari rate limit
        if batch_idx < len(batches) - 1:
            batch_wait_time = random.uniform(10, 15)  # Delay acak antar batch
            print(f"Menunggu {batch_wait_time:.1f}s sebelum melanjutkan batch berikutnya...")
            time.sleep(batch_wait_time)

    progress_bar.empty()
    status_text.text("✅ Proses Update Selesai")

    if success_tickers:
        st.success(f"{len(success_tickers)} saham berhasil diupdate.")

    if failed_tickers:
        st.warning(f"{len(failed_tickers)} saham gagal diupdate: {', '.join(failed_tickers)}")

    st.button("🔄 Refresh Halaman", on_click=lambda: st.experimental_rerun())


def load_data(ticker: str) -> pd.DataFrame:
    """Memuat data saham dari file CSV lokal."""
    filepath = os.path.join(FOLDER_OUTPUT, f"{ticker}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["date"])
        return df
    else:
        st.error(f"Data untuk {ticker} belum tersedia. Silakan lakukan update data.")
        return pd.DataFrame()


def load_and_update_data(ticker: str) -> pd.DataFrame:
    """Memuat data saham dari file lokal. Jika tidak tersedia, mengunduh dari Alpha Vantage."""
    filepath = os.path.join(FOLDER_OUTPUT, f"{ticker}.csv")

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, parse_dates=["date"])
        return df
    else:
        try:
            df = download_with_retry(ticker)
            if not df.empty:
                df = df[['4. close']]  # Mengambil hanya kolom harga penutupan
                df.columns = ['close']  # Mengganti nama kolom
                df.index.name = 'date'
                df.to_csv(filepath)
                return df
            else:
                st.error(f"Data untuk {ticker} kosong setelah didownload.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengunduh data untuk {ticker}: {e}")
            return pd.DataFrame()
