import streamlit as st
import pandas as pd
import numpy as np
import cv2
import urllib.request

st.set_page_config(page_title="Self Driving Car Demo", layout="wide")

st.title("🚗 Self Driving Car Object Detection")
st.write("Bounding Box Visualization Demo")

DATA_URL = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/labels.csv.gz"
IMAGE_ROOT = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"

@st.cache_data
def load_metadata():
    return pd.read_csv(DATA_URL)

metadata = load_metadata()

st.sidebar.header("Filter")

object_types = metadata["label"].unique()
selected_label = st.sidebar.selectbox("Select Object Type", object_types)

filtered_data = metadata[metadata["label"] == selected_label]

if len(filtered_data) == 0:
    st.error("No data found.")
    st.stop()

frame_list = filtered_data["frame"].unique()
selected_frame = st.sidebar.selectbox("Select Frame", frame_list)

@st.cache_data(show_spinner=False)
def load_image(frame):
    image_url = IMAGE_ROOT + frame
    with urllib.request.urlopen(image_url) as response:
        image = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

image = load_image(selected_frame)

# IMPORTANT FIX 👇
image = np.ascontiguousarray(image.copy())

boxes = filtered_data[filtered_data["frame"] == selected_frame]

for _, row in boxes.iterrows():
    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

st.image(image, use_container_width=True)

st.success("App Running Successfully 🎉")