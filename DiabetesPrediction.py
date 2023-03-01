#START
#import packages
import pickle
import streamlit as st
import time
from PIL import Image

#import model
diabetes_model = pickle.load(open("C:\\Users\\S P Mithun\\Downloads\\model1.pkl", 'rb'))

#create UI
st.title('Diabetes Prediction')    

Pregnancies = st.number_input('Number of Pregnancies')
        
Glucose = st.number_input('Glucose Level')
   
BloodPressure = st.number_input('Blood Pressure value')
    
SkinThickness = st.number_input('Skin Thickness value')
    
Insulin = st.number_input('Insulin Level')
    
BMI = st.number_input('BMI value')
    
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    
Age = st.number_input('Age of the Person')
    

diab_diagnosis = ''
    
if st.button('Get Result'):
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
   
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text('Getting your Result...')

    for i in range(100):

        progress_bar.progress(i + 1)

        time.sleep(0.1)

    status_text.text('Here is your Result')
       
    
    if (diab_prediction[0] == 1):
         diab_diagnosis = 'You are diabetic'
         image = Image.open("C:\\Users\\S P Mithun\\Downloads\\Diabetic.jpg")
         st.image(image)
         
    else:
         diab_diagnosis = 'You are not diabetic'
         image = Image.open("C:\\Users\\S P Mithun\\Downloads\\tips-to-prevent-diabetes.jpg")
         st.image(image)
         st.balloons()
          
        
st.success(diab_diagnosis)

#END
