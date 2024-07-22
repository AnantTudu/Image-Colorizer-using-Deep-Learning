import streamlit as st
import cv2
import numpy as np
import os

def color(image_data):
    # Load the colorization model
    DIR = r"C:\Users\tudua\Desktop\Colorizing-black-and-whiteEXP\Colorizing-black-and-white-images-using-Python-master"
    PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
    POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
    MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Convert the image to LAB color space
    scaled = image_data.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image_data.shape[1], image_data.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Convert to correct format for Streamlit
    colorized = (255 * colorized).astype("uint8")
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    return colorized

def main():
    st.title("Black & White to Color Converter")
    st.write("Upload a black and white image and convert it to color.")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_data = cv2.imdecode(file_bytes, 1)

        # Display the original image
        st.image(image_data, caption="Original Image", use_column_width=True)

        # Convert the image to color
        colorized_image = color(image_data)

        # Display the colorized image
        st.image(colorized_image, caption="Colorized Image", use_column_width=True)

if __name__ == "__main__":
    main()
