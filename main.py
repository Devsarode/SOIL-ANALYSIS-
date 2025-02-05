from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

import os, sys, glob, re

model_path = 'C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\model_soil.h5'

SoilNet = load_model(model_path)

classes = {0: "Alluvial Soil", 2: "Clay Soil ", 1: "Red Soil"}

def model_predict(image_path, model):
    print("Predicted")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))

    print(result)
    prediction = classes[result]
    print(prediction)

# Testing all photos for alluvial'
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 1.png', SoilNet)
# Add other model_predict calls here as needed

model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 2.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 3.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 4.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 5.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 6.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 7.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 8.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 9.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 10.png', SoilNet)
model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Alluvial soil\\alluvial 11.png', SoilNet)

# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 1.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 2.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 3.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 4.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 6.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 5.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 7.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 8.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 9.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 10.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 11.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 12.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 13.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 14.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 16.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 15.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 17.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 18.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 19.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 20.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 21.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 22.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 24.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 25.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 26.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 27.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Clayey soils\\clay 28.png', SoilNet)
    
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 1.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 2.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 4.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 5.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 3.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 6.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 7.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 8.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 9.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 10.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 11.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 13.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 14.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 15.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 16.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 17.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 18.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 19.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 20.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 22.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 23.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 24.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 25.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 26.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 27.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 28.png', SoilNet)
# model_predict('C:\\Users\\aman0\\OneDrive\\Desktop\\new_ML_MODEL\\SOIL_CLASS\\Soil Types\\Laterite soil\\laterite 29.png', SoilNet)

# x = open('predict.txt', 'r+')
# print(pred)
# x.writelines(pred)
# x.close()

