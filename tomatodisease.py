import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.title("IMAGE CLASSIFICATION ")
st.text("UPLOAD THE IMAGE")


uploaded_file=st.file_uploader("Choose an image",type='jpg')
if uploaded_file is not None:
  img=Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')
  model=pickle.load(open('tomato.pkl','rb'))
  if st.button('PREDICT'):
    CATEGORIES=['Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato__healthy']
    flat_data=[]
    st.write('RESULTS')
    img=np.array(img)
    img_resized=resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)
    y_out=model.predict(flat_data)
    y_out1=CATEGORIES[y_out[0]]
    st.title(f'PREDICTED OUTPUT',{y_out1})
