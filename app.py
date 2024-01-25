from flask import Flask, request, jsonify, render_template,send_from_directory
import joblib
import ast
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
app = Flask(__name__)
# Your Flask routes go here

gender_encoder = joblib.load('gender_encoder.joblib')
admit_type_encoder = joblib.load('admit_type_encoder.joblib')
admit_location_encoder = joblib.load('admit_location_encoder.joblib')
model =joblib.load('model.joblib')
X_scaled = joblib.load('X_scaled.joblib')
# @app.route('/predict/los', methods=['POST'])
# def predict():
#  #input_string = request.form.get('user_input')
#  data = request.get_json()
#  input_string = data.get('pred', 'No message received')
#  feature_vector = ast.literal_eval(input_string)[0] 
#  input = [feature_vector]
#  print(input)
#  numpy_array = np.array(input, dtype=float)
#  output = model.predict(numpy_array)
#  print(output)
#  response_data = {'status': 'success', 'prediction': output[0]}
#  return jsonify(response_data)

@app.route('/predict/los', methods=['POST'])
def predict():
    if 'file' not in request.files:
        obj = {"status": "fail", "message": 'No file part'}
        return jsonify(obj)

    file = request.files['file']

    # Validate file type
    if file.filename.endswith('.csv'):
        # Process CSV data
        df = pd.read_csv(file)
        print(df)
        # Transform the categorical variable using the loaded encoder
        categorical_columns = ['gender', 'admit_type', 'admit_location']  # Add more columns as needed
        input_to_model=[list(df.iloc[i]) for i in range(len(df))]
        for column in categorical_columns:
        # Extract the categorical variable
            categorical_data = df[column]

            # Reshape the data for the encoder
            categorical_data_reshaped = categorical_data.values.reshape(-1, 1)

            # Transform the categorical variable using the loaded encoder
            if column=='gender':
                one_hot_encoded_categorical = gender_encoder.transform(categorical_data_reshaped)
                one_hot_encoded_categorical = one_hot_encoded_categorical.toarray()
            elif column=='admit_type':
                one_hot_encoded_categorical = admit_type_encoder.transform(categorical_data_reshaped)
                one_hot_encoded_categorical = one_hot_encoded_categorical.toarray()
            else:
                one_hot_encoded_categorical = admit_location_encoder.transform(categorical_data_reshaped)
                one_hot_encoded_categorical = one_hot_encoded_categorical.toarray()

            print(one_hot_encoded_categorical)
            # Drop the original categorical column from the DataFrame
            for j in range (0, len(input_to_model)):
                for i in range(0, len(input_to_model[j])):
                    if isinstance(input_to_model[j][i], str):
                        del input_to_model[j][i]
                        break

            # Combine the one-hot encoded categorical variable with the numerical variables
            for j in range(0, len(one_hot_encoded_categorical)):
                input_to_model[j].extend(one_hot_encoded_categorical[j][i] for i in range(0, len(one_hot_encoded_categorical[0]))) 
                print(input_to_model)

        # Convert the list to a NumPy array
        input_to_model = np.array(input_to_model)
        print(input_to_model.shape)
        scaled_input = X_scaled.transform(input_to_model)
        prediction = model.predict(scaled_input)
        print(prediction)
        obj = {"status": "success","prediction": prediction.tolist()}
        return jsonify(obj)


plt.switch_backend('Agg')
@app.route('/predict/plot', methods=['POST'])
def plot():
    if 'file' not in request.files:
        obj = {"status": "fail", "message": 'No file part'}
        return jsonify(obj)

    file = request.files['file']

    # Validate file type
    if file.filename.endswith('.csv'):
        # Process CSV data
        df = pd.read_csv(file)
        # Create a simple plot
        data = df['Diseases']

        plt.bar(data.value_counts().index, data.value_counts(), color='red')
        plt.xlabel('Diseases')
        plt.ylabel('Number of Patients')
        plt.title('Number of Patients vs Diseases')

        # Convert the plot to a base64-encoded image
        plot_image_base64 = plot_to_base64(plt)

        # Close the plot to release resources
        plt.close()

        value_counts = df['Diseases'].value_counts()
        percentage_counts = value_counts / len(df['Diseases']) * 100

        unique_values = percentage_counts.index
        print(unique_values)

        # Return the base64-encoded image as part of the JSON response
        obj = {"status": "success", "location": df.loc[0, 'Location'], "image": plot_image_base64, "percentage_counts": percentage_counts.tolist(), "diseases": unique_values.tolist()}
        print(obj)
        return jsonify(obj)
    else:
        obj = {"status": "fail", "message": 'Invalid file type. Please upload a CSV file'}
        return jsonify(obj)

def plot_to_base64(plt_obj):
    # Save the plot to a BytesIO object
    plot_bytes = BytesIO()
    plt_obj.savefig(plot_bytes, format='png')
    plot_bytes.seek(0)

    # Convert the plot to a base64-encoded image
    plot_base64 = base64.b64encode(plot_bytes.read()).decode('utf-8')
    return plot_base64

if __name__ == "__main__":
  app.run() 