from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Model import create_model

# Direktori gambar dataset
test_dir = 'C:/Code/AI-Face/Fer-2013/test'
train_dir = 'C:/Code/AI-Face/Fer-2013/train'

# Tujuan Normalisasi
testData = ImageDataGenerator(rescale=1./255)
trainData = ImageDataGenerator(rescale=1./255)

# Memuat gambar dari direktori folder
TrainGen = trainData.flow_from_directory(
    train_dir,  # Path ke direktori pelatihan
    target_size=(48, 48),  # Ukuran gambar yang akan selalu sesuai 48 x 48
    batch_size=64,  # Menggunakan batch untuk efisiensi pelatihan
    color_mode="grayscale",  # Memastikan dataset FER2013 berisi gambar wajah dalam format grayscale
    class_mode='categorical'  # Mengkodekan label emosi dalam format one-hot encoding
)

TestGen = testData.flow_from_directory(
    test_dir,  # Path ke direktori pengujian
    target_size=(48, 48),  # Ukuran gambar yang akan selalu sesuai 48 x 48
    batch_size=64,  # Menggunakan batch untuk efisiensi pelatihan
    color_mode="grayscale",  # Memastikan dataset FER2013 berisi gambar wajah dalam format grayscale
    class_mode='categorical'  # Mengkodekan label emosi dalam format one-hot encoding
)

# Membuat model dan melatihnya
model = create_model()  # Ini harusnya dipanggil sebagai fungsi

# Adam adalah algoritma optimasi yang umum digunakan yang menggabungkan kelebihan dari metode AdaGrad dan RMSProp.
# Algoritma ini melakukan pembaruan bobot dengan menggunakan gradien dan mengadaptasi langkah pembaruan berdasarkan momen (momentum) dari gradien yang lebih sebelumnya.

# loss = 'categorical_crossentropy', ini adalah fungsi kerugian yang digunakan ketika kita memiliki lebih dari dua kelas dan labelnya diubah ke format one-hot encoding (misalnya, emosi marah, senang, sedih, dll.).
# Fungsi ini menghitung seberapa baik model memprediksi kelas yang benar. Semakin kecil nilai loss, semakin baik model tersebut.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # metrics accuracy berfungsi untuk memantau akurasi model selama pelatihan dan evaluasi

history = model.fit(
    TrainGen,
    # TrainGen.samples adalah jumlah total contoh dalam data pelatihan.
    # TrainGen.batch_size adalah ukuran batch yang telah ditentukan sebelumnya
    steps_per_epoch=TrainGen.samples // TrainGen.batch_size,
    # Data yang digunakan untuk memvalidasi model selama pelatihan
    validation_data=TestGen,
    # Ini serupa dengan steps_per_epoch, tetapi digunakan untuk data validasi. Ini menentukan berapa banyak langkah yang akan dilakukan untuk validasi.
    validation_steps=TestGen.samples // TestGen.batch_size,
    # Menentukan jumlah epoch yang akan dijalankan. Satu epoch berarti model telah melihat seluruh data pelatihan sekali.
    # Dalam hal ini, model akan dilatih selama 30 epoch.
    epochs=30
)

# Menyimpan model
model.save('emotion_face.h5')
