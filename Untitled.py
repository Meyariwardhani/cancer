#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Membaca model prediksi breastcancer
breastcancer_model = joblib.load('svm_model.sav')

# Membaca model clustering breastcancer (hasil PCA dengan 1 komponen utama)
clustering_model = joblib.load('kmeans_model.sav')

# Membaca model scaler
scaler = joblib.load('scaler_model2.sav')

# Judul web
st.title('Prediksi Diagnosis Kanker Payudara')


# Menampilkan hasil prediksi
radius_mean_input = st.text_input('Input nilai Radius mean')
texture_mean_input = st.text_input('Input nilai Texture mean')
perimeter_mean_input = st.text_input('Input nilai Perimeter mean')
area_mean_input = st.text_input('Input nilai Area mean')
smoothness_mean_input = st.text_input('Input nilai Smoothness mean')
compactness_mean_input = st.text_input('Input nilai Compactness mean')
concavity_mean_input = st.text_input('Input nilai Concavity mean')
concave_points_mean_input = st.text_input('Input nilai Concave points mean')
symmetry_mean_input = st.text_input('Input nilai Symmetry mean')
fractal_dimension_mean_input = st.text_input('Input nilai Fractal dimension mean')


# Validasi input
if radius_mean_input.strip() and texture_mean_input.strip() and perimeter_mean_input.strip() and area_mean_input.strip() and smoothness_mean_input.strip() and compactness_mean_input.strip() and concavity_mean_input.strip() and concave_points_mean_input.strip() and symmetry_mean_input.strip() and  fractal_dimension_mean_input.strip():
    radius_mean = float(radius_mean_input)
    texture_mean = float(texture_mean_input)
    perimeter_mean = float(perimeter_mean_input)
    area_mean = float(area_mean_input)
    smoothness_mean = float(smoothness_mean_input)
    compactness_mean = float(compactness_mean_input)
    concavity_mean = float(concavity_mean_input)
    concave_points_mean = float(concave_points_mean_input)
    symmetry_mean = float(symmetry_mean_input)
    fractal_dimension_mean = float(fractal_dimension_mean_input)

    # Code untuk prediksi
    # Membuat tombol untuk prediksi
if st.button('Test Prediksi Diagnosis Kanker Payudara'):
    input_data = np.array([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,  fractal_dimension_mean]).reshape(1, -1)
    breastcancer_prediction = breastcancer_model.predict(input_data)

    # Menampilkan hasil prediksi
    if breastcancer_prediction[0] == 1:
        breastcancer_diagnosis = 'Pasien terdiagnosis kanker ganas'
        st.success(breastcancer_diagnosis)
    else:
        breastcancer_diagnosis = 'Pasien terdiagnosis kanker jinak'
        st.error(breastcancer_diagnosis)

        # Melakukan clustering untuk penderita anemia
        # Scaling hanya pada variabel yang digunakan untuk pengklasteran
        clustering_data = np.array([radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,  fractal_dimension_mean]).reshape(1, -1)
        clustering_data_scaled = scaler.transform(clustering_data)

        breastcancer_severity = clustering_model.predict(clustering_data_pca)
        if breastcancer_severity[0] == 0:
            severity = 'Jinak'
        else:
            severity = 'Ganas'
        
        st.write(f'Tingkat keparahan Kanker payudara: {severity}')
else:
    st.warning('Mohon lengkapi semua kolom input.')




