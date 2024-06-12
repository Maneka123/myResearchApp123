from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO
from flask_cors import cross_origin
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import googlemaps
from googlemaps.exceptions import ApiError
from geopy.geocoders import Nominatim

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
import mysql.connector
import base64

app = Flask(__name__)
app.secret_key = '09cd6cb8206a12b54a7ddb28566be757'
socketio = SocketIO(app, cors_allowed_origins="*")
bcrypt = Bcrypt(app)

# Google Maps API Key
GOOGLE_MAPS_API_KEY = 'AIzaSyDa7rMGjyp1_6UI5u6F20qknR3-c0hnaog'
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

cluster = MongoClient(
        "mongodb://localhost:27017/")
db = cluster['TrustyPet']
db = cluster['TrustyPet']

fields = []
description = {}

@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/bot')
@cross_origin()
def chat_bot():
    if "email" in session:
        return render_template('bot.html')
    else:
        return redirect(url_for('login', next='/bot'))        

@app.route('/signup', methods=['POST'])
@cross_origin()
def sign_up():
    collection = db['users']
    form_data = request.form
    pw_hash =  bcrypt.generate_password_hash(form_data['password']).decode('utf-8')
    result = collection.find_one({'email': form_data['email']})
    if result == None:
        id = collection.insert_one({
            'name': form_data['name'],
            'email': form_data['email'],
            'password': pw_hash
        }).inserted_id
        response = ''
        if id == '':
            response = 'failed'
        else:
            response = 'success'
    else:
        response = 'failed'
    return render_template('signup.html', response=response)


@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == 'POST':
        collection = db['users']
        form_data = request.form
        next_url = request.form.get('next')
        query_data = {
            "email": form_data['email']
        }
        result = collection.find_one(query_data)
        response = ''
        if result == None:
            response = 'failed'
        else:
            pw = form_data['password']
            if bcrypt.check_password_hash(result['password'], pw) == True:
                response = 'succeeded'
                session['email'] = form_data['email']
                session['name'] = result['name']
                
            else:
                response = 'failed'
        
        if response == 'failed':
            return render_template('login.html', response=response)
        else:
            if next_url:
                return redirect(next_url)
            else:
                return redirect(url_for('chat_bot'))
    else:
        if "email" in session:
            return redirect(url_for("chat_bot"))
        else:
            return render_template('login.html')
        
@app.route('/dashboard')
@cross_origin()
def dashboard():
    if "email" in session:
        return render_template('dashboard.html')
    else:
        return redirect(url_for('login'))
    
@app.route('/logout')
@cross_origin()
def logout():
    if 'email' in session:
        session.pop("email", None)
        session.pop("username", None)
        
        return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))

@app.route('/contact')
@cross_origin()
def contact_us():
    return render_template('contactUs.html')

@app.route('/bot',methods=['POST'])
@cross_origin()
def bot():
   
    # Load the dataset
    data = pd.read_csv('medical_data.csv')
    symptom_features = list(set(data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].values.ravel()))
    # Convert the Symptoms column to a list of lists for one-hot encoding
    data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']] = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].apply(lambda x: x.str.split(','))
   
   # Flatten the symptom lists
    data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']] = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].apply(lambda x: [item for sublist in x for item in sublist])    
    symptoms = data[['Symptom1', 'Symptom2', 'Symptom3', 'Symptom4', 'Symptom5', 'Symptom6', 'Symptom7', 'Symptom8', 'Symptom9', 'Symptom10', 'Symptom11', 'Symptom12', 'Symptom13', 'Symptom14', 'Symptom15', 'Symptom16', 'Symptom17', 'Symptom18', 'Symptom19', 'Symptom20', 'Symptom21', 'Symptom22', 'Symptom23', 'Symptom24', 'Symptom25', 'Symptom26', 'Symptom27', 'Symptom28', 'Symptom29', 'Symptom30', 'Symptom31', 'Symptom32', 'Symptom33', 'Symptom34', 'Symptom35', 'Symptom36', 'Symptom37', 'Symptom38', 'Symptom39', 'Symptom40', 'Symptom41', 'Symptom42', 'Symptom43', 'Symptom44', 'Symptom45', 'Symptom46', 'Symptom47', 'Symptom48', 'Symptom49', 'Symptom50', 'Symptom51', 'Symptom52']].values.tolist()   
    
    # Initialize a MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(classes=symptom_features)
    mlb.fit(symptoms)
    symptoms_encoded = mlb.transform(symptoms)

    # One-hot encoded symptoms
    X = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)
    y = data['Disease']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    #Handling User Input
    user_data = request.json  
    symptoms_input = user_data.get('symptoms', "").split(',')
    print(user_data)
    
    
    # Encoding the user's symptoms using the same MultiLabelBinarizer
    user_symptoms_encoded = mlb.transform([symptoms_input])

    # One-hot encoded symptoms for prediction
    user_input = pd.DataFrame([list(user_symptoms_encoded[0])])

    # Predicting the disease using the trained model
    predicted_disease = classifier.predict(user_input)
    print(predicted_disease)
    return jsonify({'predicted_disease': predicted_disease[0]})

@app.route('/getinfo',methods=['POST'])
@cross_origin()
def getinfo():
    from urllib.parse import unquote
    
    # Accessing URL-encoded parameters from the query string
    infodisease = unquote(request.json.get('disease', "").split('=', 1)[1])
    print(infodisease)
    
    description_data = pd.read_csv('description.csv')
    treatment_data = pd.read_csv('treatment.csv')
    symptoms_data = pd.read_csv('symptoms.csv')
    prevention_data = pd.read_csv('prevention.csv')
    testing_data = pd.read_csv('testing.csv')
   
    predicted_treatment=treatment_data.loc[treatment_data['Disease'].str.strip() == infodisease]['Treatment'].values[0]
    predicted_description=description_data.loc[description_data['Disease'].str.strip() == infodisease]['Description'].values[0]
    predicted_symptoms=symptoms_data.loc[symptoms_data['Disease'].str.strip() == infodisease]['Symptoms'].values[0]
    predicted_prevention=prevention_data.loc[prevention_data['Disease'].str.strip() == infodisease]['Prevention'].values[0]
    predicted_testing=testing_data.loc[testing_data['Disease'].str.strip() == infodisease]['Testing'].values[0]

    # Construct response
    response = {
        'predicted_disease': infodisease,
        'treatment': predicted_treatment,
        'description': predicted_description,
        'symptoms': predicted_symptoms,
        'prevention': predicted_prevention,
        'testing': predicted_testing
    }

    return jsonify(response)

@app.route('/info')
def info():
    # Accessing URL-encoded parameters from the query string
    disease = request.args.get('disease')
    return render_template("info.html", disease=disease)

# Create a geocoder instance
geolocator = Nominatim(user_agent="animal_clinics_and_veterinarians")

@app.route('/find_veterinarians_and_animal_clinics', methods=['POST'])
@cross_origin()
def find_veterinarians_and_animal_clinics():
    address = request.form.get('address')
    zip_code = request.form.get('zip_code')
    location_query = f"{address}, {zip_code}"

    try:
        # Geocode the address and zip code to obtain the latitude and longitude
        location = geolocator.geocode(location_query)
        latitude = location.latitude
        longitude = location.longitude

        radius = 2000  

        # Use Google Maps API to find animal clinics within the specified radius
        animal_clinics = gmaps.places(query="animal clinic", location=(latitude, longitude), radius=radius)
        animal_clinic_results = animal_clinics['results']
        
        # Use Google Maps API to find veterinarians within the specified radius
        veterinarians = gmaps.places(query="veterinarian", location=(latitude, longitude), radius=radius)
        veterinarian_results = veterinarians['results']
        
        # Combine the results of both queries
        combined_results = animal_clinic_results + veterinarian_results
        
        return render_template('clinics_vets.html', results=combined_results, location=location_query)
    except ApiError as e:
        # Handle API errors
        return render_template('error.html', error=str(e))

########################################MANEKA#######################################





# Load the trained models
BREED_MODEL_PATH = 'breedCheckpoint\\epoch_5_checkpoint(12).pth'
HEALTH_MODEL_PATH = 'healthCheckpoint\\epoch_5_checkpoint(13).pth'
EMOTION_MODEL_PATH = 'emotionPrediction\\checkpoints\\epoch_5_checkpoint(14).pth'
AGE_MODEL_PATH = 'agePrediction\\epoch_5_checkpoint(15).pth'
GENDER_MODEL_PATH = 'genderPrediction\\inception_epoch_5(5).pth'

breed_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=10)
health_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
emotion_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
age_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
gender_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)
gender_model.fc = torch.nn.Linear(gender_model.fc.in_features, 2)  # Assuming 2 classes for gender
gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=torch.device('cpu')))

breed_model.load_state_dict(torch.load(BREED_MODEL_PATH, map_location=torch.device('cpu')))
health_model.load_state_dict(torch.load(HEALTH_MODEL_PATH, map_location=torch.device('cpu')))
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=torch.device('cpu')))
age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=torch.device('cpu')))

breed_model.eval()
health_model.eval()
emotion_model.eval()
age_model.eval()
gender_model.eval()

# Transformations
transform_new_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Class mappings
class_to_breed = {
    0: 'Abyssinian',
    1: 'American Shorthair',
    2: 'Balinese',
    3: 'Bengal',
    4: 'Birman',
    5: 'Bombay',
    6: 'British Shorthair',
    7: 'Persian',
    8: 'Siamese',
    9: 'Sphynx'
}

class_to_health = {
    0: 'have good stamina and can maintain its activity levels.',
    1: 'might not be the most energetic due to their weight',
    2: 'might need some time to bounce back to full stamina'
}

class_to_emotion = {
    0: 'Angry',
    1: 'Happy',
    2: 'Relaxed',
    3: 'Sad'
}

class_to_age = {
    0: 'Kitten - Age Range : 0-1 year old',
    1: 'Young Cat - Age Range : 1-10 years old',
    2: 'Old Cat - Age Range : 10+ years old'
}

class_to_gender = {
    0: 'Female',
    1: 'Male'
}

def predict_with_confidence(model, image, class_mapping, top_n=3):
    image = transform_new_image(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probabilities, min(top_n, len(class_mapping)))
        results = [(class_mapping[class_id.item()], prob.item()) for class_id, prob in zip(top_classes[0], top_probs[0])]
    return results

# Ensure you have the MySQL server running and the specified database and user exist.
mydb = mysql.connector.connect(
    host="localhost",
    port="3307",  # Default MySQL port
    user="root",
    password="",  # Use your MySQL root password
    database="animal_datatwo"
)

mycursor = mydb.cursor(buffered=True)

mycursor.execute("CREATE TABLE IF NOT EXISTS cats (id INT AUTO_INCREMENT PRIMARY KEY, image LONGBLOB, breed VARCHAR(100), gender VARCHAR(10), age VARCHAR(10), stamina VARCHAR(50), emotion VARCHAR(50))")

# Admin password
ADMIN_PASSWORD = "123"

def is_admin(request):
    return request.form.get("admin_password") == ADMIN_PASSWORD

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if is_admin(request):
            if 'add_cat' in request.form:
                image_file = request.files['image']
                image_data = image_file.read()
                breed = request.form['breed']
                gender = request.form['gender']
                age = request.form['age']
                stamina = request.form['stamina']
                emotion = request.form['emotion']

                sql = "INSERT INTO cats (image, breed, gender, age, stamina, emotion) VALUES (%s, %s, %s, %s, %s, %s)"
                val = (image_data, breed, gender, age, stamina, emotion)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat added successfully!', 'success')
                return redirect(url_for('index'))

            elif 'edit_cat' in request.form:
                cat_id = request.form['cat_id']
                breed = request.form['edit_breed']
                gender = request.form['edit_gender']
                age = request.form['edit_age']
                stamina = request.form['edit_stamina']
                emotion = request.form['edit_emotion']

                sql = "UPDATE cats SET breed = %s, gender = %s, age = %s, stamina = %s, emotion = %s WHERE id = %s"
                val = (breed, gender, age, stamina, emotion, cat_id)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat updated successfully!', 'success')
                return redirect(url_for('index'))

            elif 'delete_cat' in request.form:
                cat_id = request.form['cat_id']
                sql = "DELETE FROM cats WHERE id = %s"
                val = (cat_id,)
                mycursor.execute(sql, val)
                mydb.commit()
                flash('Cat deleted successfully!', 'success')
                return redirect(url_for('index'))
        else:
            flash('Unauthorized access! Please enter admin password.', 'error')
            return redirect(url_for('index'))

    mycursor.execute("SELECT * FROM cats")
    cats = mycursor.fetchall()
    cat_records = []
    for cat in cats:
        cat_record = list(cat)
        cat_image_encoded = base64.b64encode(cat[1]).decode('utf-8')
        cat_record[1] = f"data:image/jpeg;base64,{cat_image_encoded}"
        cat_records.append(cat_record)
    return render_template('index18.html', cats=cat_records)

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent.'}), 400
    image_file = request.files['image']
    try:
        image = Image.open(image_file)
        image_np = np.array(image)  # Convert PIL Image to numpy array
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Only if needed
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        cat_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalcatface_extended.xml')
        faces = cat_cascade.detectMultiScale(gray, 1.01, 3)
        num_faces_detected=len(faces)
        if num_faces_detected == 0:
            return jsonify({'error': 'No cat face detected in the image.'}), 400
        
        breed_predictions = predict_with_confidence(breed_model, image, class_to_breed, top_n=3) # changed to top_n=3
        breed_results = {f"breed_{i+1}": {"name": pred[0], "confidence": f"{pred[1]*100:.2f}%"} for i, pred in enumerate(breed_predictions)}
        health, health_confidence = predict_with_confidence(health_model, image, class_to_health)[0]
        emotion, emotion_confidence = predict_with_confidence(emotion_model, image, class_to_emotion)[0]
        age, age_confidence = predict_with_confidence(age_model, image, class_to_age)[0]
        gender, gender_confidence = predict_with_confidence(gender_model, image, class_to_gender)[0]

        return jsonify({**breed_results, 'num_faces': num_faces_detected, 'cat_detected': True,
                        'health': health, 'health_confidence': health_confidence,
                        'emotion': emotion, 'emotion_confidence': emotion_confidence,
                        'age': age, 'age_confidence': age_confidence,
                        'gender': gender, 'gender_confidence': gender_confidence}), 200
    except Exception as e:
        return jsonify({'error': 'Error processing image.'}), 500



    
#checks if the Python script is being run directly
if __name__ == '__main__':
    app.run(debug=True)



