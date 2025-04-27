from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Gunakan 'Agg' untuk non-GUI, cocok buat web server
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil inputan user
    suhu = float(request.form['suhu'])
    curah_hujan = float(request.form['curah_hujan'])
    kelembapan = float(request.form['kelembapan'])

    # Dummy model prediksi (ganti dengan model asli kalau ada)
    probabilitas_banjir = (curah_hujan * 0.4 + kelembapan * 0.3 - suhu * 0.2) / 100
    hasil = 'Banjir' if probabilitas_banjir > 0.5 else 'Tidak Banjir'

    # Buat grafik
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(np.random.rand(10), np.random.rand(10), color='blue', label='Data Latih')
    ax.scatter(0.5, probabilitas_banjir, color='red', s=100, label='Prediksi')
    ax.set_ylim(0,1)
    ax.set_title('Prediksi Banjir')
    ax.set_ylabel('Probabilitas')
    ax.legend()
    plt.tight_layout()

    # Simpan ke memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    grafik_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close(fig)

    return render_template('result.html', hasil=hasil, grafik=grafik_base64)

if __name__ == '__main__':
    app.run(debug=True)
