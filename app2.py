import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Tải mô hình đã được lưu
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Tải các LabelEncoder đã được lưu
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    age = int(request.form['age'])
    workclass = request.form['workclass']
    fnlwgt = int(request.form['fnlwgt'])
    education = request.form['education']
    education_num = int(request.form['education-num'])
    marital_status = request.form['marital-status']
    occupation = request.form['occupation']
    relationship = request.form['relationship']
    race = request.form['race']
    sex = request.form['sex']
    capital_gain = int(request.form['capital-gain'])
    capital_loss = int(request.form['capital-loss'])
    hours_per_week = int(request.form['hours-per-week'])
    native_country = request.form['native-country']

    # Tạo DataFrame từ dữ liệu đầu vào
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    # Mã hóa các biến phân loại bằng các encoder đã lưu
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    for col in categorical:
        if col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col])

    # Dự đoán thu nhập
    prediction = model.predict(input_data)

    # Tạo thông báo dựa trên dự đoán
    if prediction[0] == '>50K':
        result = "Thu nhập của bạn >50K."
    else:
        result = "Thu nhập của bạn <=50K."

    return render_template('index2.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
