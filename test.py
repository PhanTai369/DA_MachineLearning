from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Tải mô hình đã lưu
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Tải tên các cột đã được mã hóa
with open('encoded_columns.pkl', 'rb') as file:
    encoded_columns = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    age = int(request.form['age'])
    workclass = request.form['workclass']
    fnlwgt = int(request.form['fnlwgt'])
    education = request.form['education']
    education_num = int(request.form['education_num'])
    marital_status = request.form['marital_status']
    occupation = request.form['occupation']
    relationship = request.form['relationship']
    race = request.form['race']
    sex = request.form['sex']
    capital_gain = int(request.form['capital_gain'])
    capital_loss = int(request.form['capital_loss'])
    hours_per_week = int(request.form['hours_per_week'])
    native_country = request.form['native_country']

    # Tạo DataFrame cho dữ liệu đầu vào
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education_num': [education_num],
        'marital_status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital_gain': [capital_gain],
        'capital_loss': [capital_loss],
        'hours_per_week': [hours_per_week],
        'native_country': [native_country]
    })

    # Tiền xử lý dữ liệu giống như trong bước huấn luyện
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for col in categorical_cols:
        input_data[col] = input_data[col].astype('category')

    input_encoded = pd.get_dummies(input_data, drop_first=True)

    # Đảm bảo rằng các cột mã hóa khớp với các cột đã lưu
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Dự đoán
    prediction = model.predict(input_encoded)

    # Chuyển đổi dự đoán thành dạng dễ đọc
    output = '>50K' if prediction[0] == 1 else '<=50K'

    return render_template('result.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
