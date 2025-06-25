import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Memuat Dataset
df = pd.read_csv('../healthcare_stroke_dataset_raw.csv')

# Menghapus kolom id
df = df.drop(columns=['id'], axis=1)

# Menghapus data sampel dengan kategori gender Other
df = df[df['gender'] != 'Other']

# Menangani missing_value pada kolom bmi dengan median
bmi_median = df['bmi'].median()
df['bmi'] = df['bmi'].fillna(bmi_median)

# Encoding variabel biner
gender_mapping = {'Male': 1, 'Female': 0}
ever_married_mapping = {'Yes': 1, 'No': 0}
residence_type_mapping = {'Urban': 1, 'Rural': 0}

df['gender'] = df['gender'].map(gender_mapping)
df['ever_married'] = df['ever_married'].map(ever_married_mapping)
df['Residence_type'] = df['Residence_type'].map(residence_type_mapping)

# Encoding variabel kategorikal
df = pd.concat([df, pd.get_dummies(df['work_type'], dtype=int)], axis=1)
df = pd.concat([df, pd.get_dummies(df['smoking_status'], dtype=int)], axis=1)
df = df.drop(columns=['work_type', 'smoking_status'], axis=1)

# Resampling dengan Over Sampling
X = df.drop(columns='stroke', axis=1)
y = df['stroke']

sampler = RandomOverSampler(sampling_strategy='minority', random_state=252)
X_res, y_res = sampler.fit_resample(X, y)

# Menggabungkan kembali data yang telah di-resample
df_preprocess = pd.concat([X_res, y_res], axis=1)

# Export ke file csv
df_preprocess.to_csv('healthcare_stroke_dataset_preprocessing.csv', index=False)

print("Preprocessing selesai. File 'healthcare_stroke_dataset_preprocessing.csv' telah berhasil dibuat.")