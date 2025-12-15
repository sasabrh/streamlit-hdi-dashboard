import streamlit as st
import pandas as pd
import altair as alt

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis IPM, TPT, dan Pendidikan Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Nama File CSV yang Diunggah ---
CSV_FILE = "HDI Unemployment and Education - Data Indonesia Provinces (2017-2023).csv"

# --- Fungsi untuk Memuat Data (Caching) ---
@st.cache_data
def load_data(file_path):
    """Memuat data dari file CSV dan melakukan preprocessing awal."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        # Bersihkan spasi di nama kolom
        df.columns = df.columns.str.strip() 
        return df
    except FileNotFoundError:
        st.error(f"Error: File CSV '{file_path}' tidak ditemukan.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return pd.DataFrame()

df = load_data(CSV_FILE)

if not df.empty:
    
    # --- Preprocessing Data ---
    # Pastikan kolom tahun adalah integer
    df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce', downcast='integer')
    # Ambil kolom TPT Februari sebagai fokus (bisa diganti dengan TPT Agustus)
    tpt_kolom = 'Tingkat Pengangguran Terbuka (TPT) - Februari'
    
    # --- Judul Dashboard ---
    st.title("🇮🇩 Analisis Data Pembangunan Manusia (IPM) & Ketenagakerjaan")
    st.markdown("Analisis Tren IPM, Pengangguran, dan Pendidikan Antar Provinsi (2017-2023)")
    
    # --- Sidebar untuk Filter ---
    st.sidebar.header("Opsi Filter")

    # Filter Tahun
    min_tahun = int(df['Tahun'].min())
    max_tahun = int(df['Tahun'].max())
    
    tahun_pilihan = st.sidebar.slider(
        "Pilih Rentang Tahun:",
        min_value=min_tahun,
        max_value=max_tahun,
        value=(min_tahun, max_tahun)
    )
    
    # Filter Provinsi
    provinsi_unik = ['Semua Provinsi'] + sorted(list(df['Provinsi'].unique()))
    provinsi_pilihan = st.sidebar.multiselect(
        "Pilih Provinsi:", 
        provinsi_unik,
        default=provinsi_unik[1:] # Pilih semua provinsi secara default
    )
    
    # Terapkan Filter
    df_filtered = df[
        (df['Tahun'] >= tahun_pilihan[0]) & 
        (df['Tahun'] <= tahun_pilihan[1])
    ]
    
    if 'Semua Provinsi' not in provinsi_pilihan:
        df_filtered = df_filtered[df_filtered['Provinsi'].isin(provinsi_pilihan)]

    
    st.subheader(f"Data Hasil Filter ({len(df_filtered)} Baris)")

    # Tampilkan Data Filtered
    st.dataframe(df_filtered.tail(5), use_container_width=True) 

    st.markdown("---")

    # --- KPI Utama (Menggunakan data tahun terbaru yang tersedia dalam filter) ---
    st.header("🔑 Key Performance Indicators (Tahun Terbaru dalam Filter)")
    
    # Data tahun terbaru
    tahun_terbaru = df_filtered['Tahun'].max()
    df_latest = df_filtered[df_filtered['Tahun'] == tahun_terbaru]
    
    if not df_latest.empty:
        # Hitung rata-rata nasional dari data yang difilter
        avg_ipm = df_latest['Indeks Pembangunan Manusia'].mean()
        avg_tpt = df_latest[tpt_kolom].mean()
        avg_sekolah = df_latest['Rata-rata Lama Sekolah'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"Rata-rata IPM Nasional ({tahun_terbaru})", 
                value=f"{avg_ipm:.2f}",
                delta="IPM adalah indeks komposit" # Tambahkan konteks singkat
            )
        with col2:
            st.metric(
                label=f"Rata-rata TPT Nasional ({tahun_terbaru})", 
                value=f"{avg_tpt:.2f} %"
            )
        with col3:
            st.metric(
                label=f"Rata-rata Lama Sekolah ({tahun_terbaru})", 
                value=f"{avg_sekolah:.2f} Tahun"
            )
    
    st.markdown("---")

    # --- Visualisasi Data ---
    
    # 1. Tren IPM dari Waktu ke Waktu (Line Chart)
    st.header("📈 Tren IPM Antar Provinsi")
    
    # Agregasi data untuk tren
    # Jika banyak provinsi dipilih, kelompokkan per tahun untuk melihat rata-rata/median nasional
    df_tren = df_filtered.groupby(['Tahun', 'Provinsi'])['Indeks Pembangunan Manusia'].mean().reset_index()
    
    chart_ipm = alt.Chart(df_tren).mark_line(point=True).encode(
        x=alt.X('Tahun:O', title='Tahun'), # O untuk Ordinal
        y=alt.Y('Indeks Pembangunan Manusia:Q', title='Nilai IPM'), # Q untuk Quantitative
        color='Provinsi:N', # PERBAIKAN: string ditutup dan tipe data Altair ditambahkan
        tooltip=['Tahun', 'Provinsi', alt.Tooltip('Indeks Pembangunan Manusia', format='.2f')]
    ).properties(
        title='Tren Indeks Pembangunan Manusia per Provinsi'
    ).interactive() 
    
    st.altair_chart(chart_ipm, use_container_width=True)
    
    
    # 2. Hubungan antara IPM dan Pengangguran (Scatter Plot)
    st.header("🎯 Hubungan IPM vs. Tingkat Pengangguran Terbuka")
    
    # Gunakan data tahun terbaru untuk scatter plot
    df_scatter = df_latest[[
        'Provinsi', 
        'Indeks Pembangunan Manusia', 
        tpt_kolom, 
        'Rata-rata Lama Sekolah'
    ]].copy()
    
    scatter_plot = alt.Chart(df_scatter).mark_circle(size=60).encode(
        x=alt.X(tpt_kolom, title='TPT (%)'),
        y=alt.Y('Indeks Pembangunan Manusia', title='IPM'),
        color=alt.Color('Rata-rata Lama Sekolah', scale=alt.Scale(range='heatmap'), title='Rata-rata Sekolah'),
        tooltip=['Provinsi', alt.Tooltip(tpt_kolom, format='.2f'), alt.Tooltip('Indeks Pembangunan Manusia', format='.2f'), alt.Tooltip('Rata-rata Lama Sekolah', format='.2f')]
    ).properties(
        title=f'IPM vs. TPT per Provinsi ({tahun_terbaru})'
    ).interactive()
    
    st.altair_chart(scatter_plot, use_container_width=True)
    

else:
    st.title("🚨 Dashboard Tidak Dapat Dimuat")
    st.error(f"Pastikan file '{CSV_FILE}' ada di direktori yang sama dan formatnya benar.")