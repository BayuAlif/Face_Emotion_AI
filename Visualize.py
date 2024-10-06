import matplotlib.pyplot as plt

def visualize_result(history):
    
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    #plt.plot(history.history['accuracy'], label='Train Accuracy'):

    #Fungsi: Menggambar kurva akurasi pelatihan model selama setiap epoch.
    #history.history['accuracy']: Ini adalah daftar yang berisi nilai akurasi pelatihan untuk setiap epoch, yang dihasilkan selama pelatihan model. Ini menunjukkan seberapa baik model beradaptasi dengan data pelatihan.
    #plt.plot(history.history['val_accuracy'], label='Validation Accuracy'):

    #Fungsi: Menggambar kurva akurasi validasi model selama setiap epoch.
    #history.history['val_accuracy']: Ini adalah daftar yang berisi nilai akurasi untuk data validasi. Ini digunakan untuk mengevaluasi seberapa baik model generalisasi pada data yang tidak terlihat.
    #plt.title('Model Accuracy'):

    #Fungsi: Memberi judul pada grafik. Judul ini menjelaskan bahwa grafik ini menunjukkan akurasi model.
    #plt.ylabel('Accuracy'):

    #Fungsi: Menentukan label untuk sumbu Y. Dalam hal ini, kita memberi label "Accuracy" untuk menunjukkan bahwa sumbu ini mewakili akurasi model.
    #plt.xlabel('Epoch'):

    #Fungsi: Menentukan label untuk sumbu X. Label ini menunjukkan bahwa sumbu ini mewakili jumlah epoch.
    #plt.legend():

    #Fungsi: Menampilkan legenda pada grafik, sehingga kita dapat membedakan antara akurasi pelatihan dan akurasi validasi berdasarkan label yang ditentukan sebelumnya.
    #plt.show():

    #Fungsi: Menampilkan grafik yang telah dibuat di jendela visualisasi. Ini adalah langkah terakhir untuk menampilkan grafik pada layar.

    # Visualisasi loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    

    #Fungsi: Menggambar kurva kerugian pelatihan model selama setiap epoch.
    #history.history['loss']: Ini adalah daftar yang berisi nilai kerugian (loss) pelatihan untuk setiap epoch. Kerugian menunjukkan seberapa buruk model dalam memprediksi label data pelatihan. Semakin rendah nilai loss, semakin baik.
    #plt.plot(history.history['val_loss'], label='Validation Loss'):

    #Fungsi: Menggambar kurva kerugian validasi model selama setiap epoch.
    #history.history['val_loss']: Ini adalah daftar yang berisi nilai kerugian untuk data validasi. Ini menunjukkan seberapa baik model berfungsi pada data yang tidak terlihat.
    #plt.title('Model Loss'):

    #Fungsi: Memberi judul pada grafik kerugian.
    #plt.ylabel('Loss'):

    #Fungsi: Menentukan label untuk sumbu Y yang menunjukkan kerugian model.
    #plt.xlabel('Epoch'):

    #Fungsi: Menentukan label untuk sumbu X yang menunjukkan jumlah epoch.
    #plt.legend():

    #Fungsi: Menampilkan legenda pada grafik, sehingga kita dapat membedakan antara kerugian pelatihan dan kerugian validasi berdasarkan label yang ditentukan sebelumnya.
    #plt.show():

    #Fungsi: Menampilkan grafik kerugian pada layar.