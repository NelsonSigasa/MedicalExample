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
