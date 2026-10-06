# TODO: WIP
# Here model is loaded and used to predict the class of the input data
# The model is loaded from the file 'model.pkl' using joblib.load() function
# The input data is passed to the model's predict() method to get the predicted class
import joblib

model = joblib.load("botnet_detector.pkl")
