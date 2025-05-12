Dataset: Pima Indians Diabetes Dataset
Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

Target Classification (Outcome):
0 = No Diabetes
1 = Diabetes

Step:
[1]. Data Preprocessing:

- Find Zero value in Glucose, Blood Pressure, Insulin, etc and set as Missing value (NaN)
- Replace NaN (zero value) with mean (menghindari bias)
- Data dibagi jadi 80% training, 20% testing
- KNN dilakukan standarisasi fitur pakai StandardScaler

[2]. Model 1. Logistic Regression

- Model Linier untuk klasifikasi biner, digunakan tanpa scaling karena robust terhadap skala dalam case sederhana
- Hasil Evaluasi:
  - Akurasi: 70.1%
  - Precision Class 1 (Diabetes): 0.59
  - Recall Class 1: 0.50
  - F1-Score Class 1: 0.54
  - Confusion Matrix:
- Interpretasi:
  - Model lebih akurat mendeteksi pasien tidak diabetes. ini disebabkan karena data yang tidak balance, karena memang pada umumnya jumlah orang yang tidak diabetes lebih banyak daripada yang diabetes
  - Kinerja terhadap pasien positif kurang optimal, (Recall = 0.50)

[3]. Model 2. K-Nearest Neighbors (KNN)

- KNN pakai parameter terbaik hasil tuning GridSearchCV, k = 19
- KNN Klasifikasi berdasarkan neighbors terdekat di bagian fitur
- Hasil Evaluasi:
  - Akurasi: 72.0%
  - Precision Class 1 (Diabetes): 0.68
  - Recall Class 1: 0.39
  - F1-Score Class 1: 0.49
  - Confusion Matrix:
- Interpretasi:
  - KNN Akurasinya lebih tingi dikit daripada logistic regression. namun recall buat class diabetes lebih rendah (0.39), artinya menandakan KNN lebih sering melewatkan kasus yg positif diabetes (False negative tinggi)
  - Precision Class 1 cukup baik (0.68), artinya saat KNN prediksi seseorang diabetes, cukup bisa dipercaya (sedikit ya)

[4]. Conclusion

- Model ini memiliki kinerja mirip2, dari segi akurasi (70% dan 72%)
- Logistic Regression lebih seimbang dalam mengenali pasien diabetes (Recall 0.50 vs 0.39)
- KNN Lebih konservatif, lebih sering kyk klasifikasi pasienny sebagai "tidak diabetes" yang bkin recallnya rendah (0.39)
- Biasanya recall pada class 1 (diabetes) sangat penting, karena klo salah klasifikasi bisa menghambat deteksi dini
- Logistic Regression lebih direkomendasikan walaupun akurasinya sedikit lebih rendah daripada KNN, karena lebih sensitive pada kasus diabetes

Notes:

- Accuracy
- Precison (1): Ukur seberapa akurat model saat prediksi seseorang positif diabetes (1). ex: model prediksi 10 org diabetes, eh trnyta hanya 7 dari 10 org yg diabetes, maka precision = 0.7
- Recall (1): Ukur seberapa baik model dalam menangkap semua kasus diabetes. ex: ada 100 org, model hanya prediksi 50 org dengan benar sebagai diabetes maka recall = 0.5
- F1-Score: rata2 harmonis dari precision dan recall, ingin keseimbangan antara precision and recall
- Kenapa class 1 yg ditekankan: lebih penting diperhatikan karena ingin meminimalkan false negative,(kyk org seharusny diabetes eh malah ga kedeteksi) (recall, precision)
