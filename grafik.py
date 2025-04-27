import matplotlib.pyplot as plt

def buat_grafik(tahun, aktual, prediksi, save_path):
    plt.figure(figsize=(10, 6))
    
    # Data Aktual
    plt.scatter(tahun, aktual, color='blue', label='Data Aktual', s=100)
    
    # Data Prediksi
    plt.scatter(tahun, prediksi, color='red', label='Prediksi ANN', s=100)
    
    # Label angka
    for x, y in zip(tahun, prediksi):
        plt.text(x, y, f'{y:.0f}', ha='center', va='bottom', fontsize=9, color='black')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Pelanggan')
    plt.title('Hasil Prediksi ANN vs Data Aktual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
