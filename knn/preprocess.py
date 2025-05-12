import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv")

# Drop 'Outcome' karena itu label, bukan fitur
features = df.drop(columns=["Outcome"])

# Pisahkan berdasarkan label diabetes
diabetic = df[df["Outcome"] == 1].drop(columns=["Outcome"])  #diabetes
non_diabetic = df[df["Outcome"] == 0].drop(columns=["Outcome"])  #non-diabetes

# Hitung rata-rata tiap fitur untuk masing-masing
diabetic_avg = diabetic.mean(numeric_only=True).to_dict()
non_diabetic_avg = non_diabetic.mean(numeric_only=True).to_dict()

df_avg = pd.DataFrame([diabetic_avg, non_diabetic_avg], index=["Diabetic", "Non-Diabetic"])
df_avg.to_csv("preprocessed_data.csv")

print("Preprocessing selesai! Data rata-rata sudah disimpan.")
