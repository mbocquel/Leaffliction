import streamlit as sl
import requests

sl.markdown("""<style>
            h1 {
                text-align:center;
            }
            .st-emotion-cache-13ln4jf {
                padding:1rem 1rem 10rem;
            }
            .st-emotion-cache-12fmjuu.ezrtsby2 {
                display: none;
                height: 0;
            }
            </style>""", unsafe_allow_html=True)

sl.markdown("<h1>Leaffliction</h1>", unsafe_allow_html=True)

intro = """
<p>
    Welcome to my leaves recognition program. 
    The program uses a Convolutional Neural Networks to identify the category of a 
    leave form the following categories : 
</p>"""

sl.markdown(intro, unsafe_allow_html=True)

col1, col2, col3, col4 = sl.columns(4)
col1.markdown("**Apple Black_rot**")
col1.markdown("**Apple healthy**")
col2.markdown("**Apple rust**")
col2.markdown("**Apple scab**")
col3.markdown("**Grape Black rot**")
col3.markdown("**Grape Esca**")
col4.markdown("**Grape healthy**")
col4.markdown("**Grape spot**")

def sendData(img):
    result = requests.post("http://backend_fastapi:8888/predict", files={"file":img})
    return result.status_code, result.json()

result_req = None
status_code = 0

with sl.form("Try the program !", clear_on_submit=True):
    image = sl.file_uploader("Upload a leaf photo to categorise", type=["png", "jpg", "jpeg"])
    s_state = sl.form_submit_button("Send !")
    if s_state:
        if image is None:
            sl.warning("Please select a photo")
        else:
            sl.success("Information sent")
            status_code, result_req = sendData(image)
            

if status_code == 200:
    sl.markdown("## Prediction")
    left_co, cent_co,last_co = sl.columns(3)
    with cent_co:
        sl.image(image)
    prediction = result_req["prediction"]
    confidence = str(float(result_req["confidence"])*100)
    sl.markdown(f"Our model predict that it is an **{prediction}** with a confidence of **{confidence}%**")

