from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential()

    # Lapisan konvolusi pertama
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Lapisan konvolusi kedua
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Lapisan konvolusi ketiga
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Mengubah bentuk ke 1D
    model.add(Flatten())

    # Lapisan Dense (fully connected)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Lapisan output
    model.add(Dense(7, activation='softmax'))  # Sesuaikan dengan jumlah kelas emosi

    return model
