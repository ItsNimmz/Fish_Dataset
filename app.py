from flask import Flask, render_template, request
import joblib

# Load the trained model
clf = joblib.load('fish_species_classifier.pkl')

app = Flask(__name__)

# Define a route to handle the form input
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        length1 = float(request.form['length1'])
        length2 = float(request.form['length2'])
        length3 = float(request.form['length3'])
        height = float(request.form['height'])
        width = float(request.form['width'])
        
        # Predict the species using the loaded model
        input_data = [[length1, length2, length3, height, width]]
        predicted_species = clf.predict(input_data)[0]
        
        return render_template('result.html', species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
