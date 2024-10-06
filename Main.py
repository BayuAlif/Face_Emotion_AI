from Predict import detect_emotion_from_webcam

if __name__ == "__main__":
    model_path = 'C:\\Code\\AI-Face\\emotion_face.h5'  # Pastikan model berada di direktori yang sama
    class_labels = ['marah', 'senang', 'sedih', 'takut', 'jijik', 'netral', 'bingung']  # Sesuaikan dengan label yang Anda latih  # Sesuaikan dengan label yang Anda latih
    
    # Panggil fungsi deteksi emosi dari webcam
    detect_emotion_from_webcam(model_path, class_labels)
