import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import warnings

# Mengabaikan warning yang sering muncul dari statsmodels/linearmodels saat modeling
warnings.filterwarnings("ignore")

def load_and_prepare_data(file_path):
    """
    Memuat data, membersihkan nama kolom, dan menyiapkan MultiIndex Panel Data.
    """
    # 1. Memuat Data
    df = pd.read_csv(file_path)

    # 2. Membersihkan dan Mengganti Nama Kolom
    df.columns = [
        'Provinsi', 'TPT_Feb', 'TPT_Ags', 'TPAK_Feb', 'TPAK_Ags', 
        'Tahun', 'IPM', 'Rata_Rata_Lama_Sekolah'
    ]

    # Pilih hanya kolom yang dibutuhkan untuk modeling
    df = df[[
        'Provinsi', 'Tahun', 'IPM', 'TPT_Ags', 'TPAK_Ags', 'Rata_Rata_Lama_Sekolah'
    ]].copy()

    # 3. Mengubah 'Tahun' menjadi Tipe Data Waktu
    df['Tahun'] = pd.to_datetime(df['Tahun'], format='%Y')

    # 4. Membuat MultiIndex: [Entitas, Waktu]
    # Struktur ini menggabungkan Provinsi (Entitas) dan Tahun (Waktu) sebagai indeks unik.
    # Ini wajib agar Python tahu bahwa data ini adalah Panel Data.
    df = df.set_index(['Provinsi', 'Tahun'])
    
    # Visualisasi konsep: MultiIndex
    print("\n") 
    print("✅ Data Siap Model (MultiIndex Dibuat)")
    return df

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
    fe_model = PanelOLS(
        dependent_var, 
        exog, 
        # entity_effects=True: Mengontrol semua faktor unik antar-provinsi yang tidak berubah (e.g., geografi, budaya).
        entity_effects=True,  
        # time_effects=True: Mengontrol semua tren yang terjadi di tahun tertentu dan memengaruhi semua provinsi (e.g., krisis nasional).
        time_effects=True     
    )
    
    # Visualisasi konsep: Perbedaan Fixed Effects (FE) vs. Pooled OLS (regresi biasa)
    print("\n")
    print("FE menggeser garis regresi untuk setiap entitas, mengendalikan perbedaan dasar antar provinsi.")

    # Fit model dan tampilkan hasilnya
    fe_results = fe_model.fit()
    
    return fe_results

# --- EXECUTION ---

# Tentukan jalur file
file_path = 'HDI Unemployment and Education - Data Indonesia Provinces (2017-2023).csv'

# Panggil fungsi persiapan data
df_prepared = load_and_prepare_data(file_path)

# Panggil fungsi pemodelan dan dapatkan hasilnya
print("\n--- Hasil Model Fixed Effects (Efek Tetap Provinsi & Waktu) ---")
results = run_fixed_effects_model(df_prepared)
print(results)