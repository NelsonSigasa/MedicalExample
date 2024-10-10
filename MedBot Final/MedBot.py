import os
import cv2
import time
import random
import numpy as np
import pandas as pd
from gtts import gTTS
from queue import Queue
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import speech_recognition as sr

# Sample data for machine learning
data = {
    "symptoms": [
        "fever", "cough", "chest pain", "shortness of breath", "fatigue",
        "fever and cough", "cough and chest pain", "fever and chest pain"
    ],
    "disease": [
        "Bacterial Infections", "Asthma", "Hypertension", "COPD", "Anxiety",
        "Bacterial Infections", "Asthma", "Hypertension"
    ]
}
##Rori


df = pd.DataFrame(data)

# Handle class imbalance
X = df['symptoms'].values
y = df['disease'].values

# Use SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X.reshape(-1, 1), y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, zero_division=1))

# Function to predict disease
def predict_disease(symptoms):
    prediction = model.predict([symptoms])
    return prediction

# Mapping of symptoms to possible diseases
symptoms_to_diseases = {
    "fever": ["Bacterial Infections", "Type 2 Diabetes"],
    "cough": ["Asthma", "Chronic Obstructive Pulmonary Disease (COPD)"],
    "chest pain": ["Hypertension", "Gastroesophageal Reflux Disease (GERD)"],
    "shortness of breath": ["Asthma", "Chronic Obstructive Pulmonary Disease (COPD)"],
    "fatigue": ["Depression", "Type 2 Diabetes", "Anxiety"]
}
##Nelson

# Dictionary of diseases and medications
medication_prescriptions = {
    "Hypertension": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide"],
    "Type 2 Diabetes": ["Metformin", "Glimepiride", "Insulin", "Sitagliptin"],
    "Asthma": ["Albuterol (inhaler)", "Fluticasone (inhaler)", "Montelukast", "Budesonide"],
    "Chronic Obstructive Pulmonary Disease (COPD)": ["Tiotropium", "Salmeterol", "Roflumilast", "Albuterol (inhaler)"],
    "Anxiety": ["Alprazolam", "Lorazepam", "Diazepam", "Buspirone"]
}

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("output.mp3")
        os.system('start output.mp3')  # Use 'open' for macOS or 'xdg-open' for Linux.
        time.sleep(10)
    except Exception as e:
        print(f"Error in speak function: {e}")

def ai_receptionist_greeting():
    speak("Hello! Welcome to the hospital. I am your virtual assistant. How can I assist you today?")
    print("AI Receptionist: Hello! Welcome to the hospital. I am your virtual assistant.")
    print("How can I assist you today?")

def generate_ticket_number():
    return random.randint(1, 201)

def print_receipt(patient_name, doctor_name, ticket_number):
    receipt = f"""
    --- Appointment Receipt ---
    Ticket Number: {ticket_number}
    Patient Name: {patient_name}
    Specialist: {doctor_name}
    ---------------------------
    """
    print(receipt)
    speak(receipt)

def book_appointment():
    global user_name

    speak("Please provide your full name:")
    user_name = input("AI Receptionist: Please provide your full name: ")

    speak("Now provide your preferred date and time:")
    appointment_date_time = input("AI Receptionist: Please provide your preferred date and time (e.g., tomorrow at 10:00 AM): ")

    speak("Please select the type of doctor or specialist you would like to see by typing the corresponding number:")
    print("1. General Practitioner")
    print("2. Cardiologist")
    print("3. Dentist")
    print("4. Optometrist")
    print("5. Gynecologist")

    specialty_choice = input("AI Receptionist: Enter the number corresponding to your choice (1-5): ")

    specialty_map = {
        '1': 'General Practitioner',
        '2': 'Cardiologist',
        '3': 'Dentist',
        '4': 'Optometrist',
        '5': 'Gynecologist'
    }
    ##Chasco
   