import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import warnings
# Tambahkan library visualisasi untuk EDA (Heatmap Korelasi)
import matplotlib.pyplot as plt
import seaborn as sns 

# Mengabaikan warning yang sering muncul dari statsmodels/linearmodels saat modeling
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """
    Memuat data, membersihkan nama kolom, dan menyiapkan MultiIndex Panel Data.
    """
    # 1. Memuat Data
    df = pd.read_csv(file_path)

    # 2. Membersihkan dan Mengganti Nama Kolom
    # Menggunakan nama kolom yang sama dengan analisis_hdi.py untuk konsistensi
    df.columns = [
        'Provinsi', 'TPT_Feb', 'TPT_Ags', 'TPAK_Feb', 'TPAK_Ags', 
        'Tahun', 'Indeks Pembangunan Manusia', 'Rata-rata Lama Sekolah'
    ]

    # Pilih hanya kolom yang dibutuhkan untuk modeling
    df = df[[
        'Provinsi', 'Tahun', 'Indeks Pembangunan Manusia', 'TPT_Ags', 'TPAK_Ags', 'Rata-rata Lama Sekolah'
    ]].copy()

    # 3. Mengubah 'Tahun' menjadi Tipe Data Waktu
    # Mengubah nama kolom untuk Linearmodels (tidak bisa ada spasi di nama kolom eksogen/endogen)
    df.rename(columns={
        'Indeks Pembangunan Manusia': 'IPM', 
        'Rata-rata Lama Sekolah': 'Rata_Rata_Lama_Sekolah',
        'TPT_Ags': 'TPT_Ags',
        'TPAK_Ags': 'TPAK_Ags'
    }, inplace=True)

    df['Tahun'] = pd.to_datetime(df['Tahun'], format='%Y')

    # 4. Membuat MultiIndex: [Entitas, Waktu]
    # Struktur ini menggabungkan Provinsi (Entitas) dan Tahun (Waktu) sebagai indeks unik.
    df = df.set_index(['Provinsi', 'Tahun'])
    
    print("\n✅ Data Siap Model (MultiIndex Dibuat)")
    return df

def generate_correlation_heatmap(df_panel):
    """
    Menghitung dan memvisualisasikan matriks korelasi untuk variabel-variabel utama.
    """
    print("\n--- Heatmap Korelasi (Untuk Cek Multikolinearitas) ---")
    
    # Ambil variabel yang sudah dinamai ulang
    corr_vars = df_panel[['IPM', 'TPT_Ags', 'TPAK_Ags', 'Rata_Rata_Lama_Sekolah']]
    
    correlation_matrix = corr_vars.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix, 
        annot=True,              
        fmt=".2f",               
        cmap='coolwarm',         
        linewidths=.5,           
        cbar_kws={'label': 'Koefisien Korelasi'}
    )
    
    plt.title('Heatmap Korelasi Variabel IPM dan Faktornya')
    plt.show() # Menampilkan plot

def run_fixed_effects_model(df_panel):
    """
    Menjalankan Model Regresi Fixed Effects (Efek Tetap Provinsi & Waktu).
    """
    # 5. Mendefinisikan Variabel
    dependent_var = df_panel['IPM']

    exog_vars = ['TPT_Ags', 'TPAK_Ags', 'Rata_Rata_Lama_Sekolah']
    
    # Menambahkan konstanta (intercept) untuk persamaan regresi
    exog = sm.add_constant(df_panel[exog_vars], prepend=False) 

    # 6. Menjalankan Model Panel OLS (dengan Efek Tetap)
    # INDENTASI SUDAH DIPERBAIKI DI SINI
    fe_model = PanelOLS(
        dependent_var, 
        exog, 
        entity_effects=True,  
        time_effects=True     
    )
    
    fe_results = fe_model.fit()
    
    return fe_results

# --- EXECUTION ---

# Tentukan jalur file
file_path = 'HDI Unemployment and Education - Data Indonesia Provinces (2017-2023).csv'

# Panggil fungsi persiapan data
df_prepared = load_and_prepare_data(file_path)

# Panggil fungsi heatmap
generate_correlation_heatmap(df_prepared) 

# Panggil fungsi pemodelan dan dapatkan hasilnya
print("\n--- Hasil Model Fixed Effects (Efek Tetap Provinsi & Waktu) ---")
results = run_fixed_effects_model(df_prepared)
print(results)