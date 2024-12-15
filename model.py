from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Kaydedilmiş modeli yükle
model = joblib.load('linear_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Formdan gelen veriyi al
            hours = float(request.form['hours'])
            scores = float(request.form['scores'])
            activities = 1 if request.form['activities'] == 'Yes' else 0  # 'Yes' için 1, 'No' için 0
            sleep = float(request.form['sleep'])
            papers = float(request.form['papers'])
            
            # Model için uygun veri formatını hazırla
            input_data = pd.DataFrame([[hours, scores, sleep, papers, activities]],
                                      columns=['Hours Studied', 'Previous Scores', 'Sleep Hours',
                                               'Sample Question Papers Practiced', 'Extracurricular Activities_Yes'])

            # Veriyi doğru formatta dönüştür
            input_data = pd.get_dummies(input_data, drop_first=True)

            # Tahmin yap
            prediction = model.predict(input_data)
            result = round(prediction[0], 2)
        except Exception as e:
            result = f"Hata: {e}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
