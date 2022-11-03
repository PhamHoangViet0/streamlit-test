import streamlit as st
import cv2 as cv
from PIL import Image
import numpy as np
import values
import dir_check

def myFunction():
    if dir_check.bardet == None:
        dir_check.init()


values.m += 1
st.title(f'Hello World!x{values.m}')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Input', use_column_width=True)

    img_array = np.array(image)[:,:,::-1]
    result = dir_check.decode(img_array)
    st.text(f'Execution time: {result[0]}')
    st.text(f'Decoded: {result[1]}')
    st.text(f'Correction value: {result[2]}')
    # cv.imshow('asd', img_array)
    # cv.waitKey(0)
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)

    # image = cv.imread(uploaded_file)
    # st.image(image, caption='Input', use_column_width=True)
    # img_array = np.array(image)
    # cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':    #added
    myFunction()    #added