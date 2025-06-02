import cv2
import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import pandas as pd
from streamlit_option_menu import option_menu




if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# --------------- Handle Query Parameters -----------------
query_params = st.query_params
if "auth" in query_params and query_params["auth"] == "True":
    st.session_state.authenticated = True
    st.session_state.user_id = query_params.get("user", "")

# --------------- Ensure User File Exists -----------------
USER_DATA_FILE = "user.csv"
if not os.path.exists(USER_DATA_FILE):
    pd.DataFrame(columns=["Email", "Pass"]).to_csv(USER_DATA_FILE, index=False)

# --------------- AUTHENTICATION SECTION -----------------



if not st.session_state.authenticated:
    st.sidebar.title("FaceMak detection")
    selected2 = st.sidebar.selectbox("LogIn or SignUp", ["LOG IN", "SIGN UP"])

    if selected2 == "SIGN UP":
        st.header("Sign Up")
        with st.form(key="signup_form"):
            email1 = st.text_input("Enter the E-MAIL")
            password1 = st.text_input("Enter the PASSWORD", type="password")
            if st.form_submit_button("SIGN UP"):
                user_file = pd.read_csv(USER_DATA_FILE)
                if email1 in user_file["Email"].values:
                    st.error("Email already exists! Please log in.")
                else:
                    new_user = pd.DataFrame([[email1, password1]], columns=["Email", "Pass"])
                    new_user.to_csv(USER_DATA_FILE, mode='a', header=False, index=False)
                    st.success("Successfully signed up! Please log in.")

    if selected2 == "LOG IN":
        st.header("Login")
        with st.form(key="login_form"):
            email = st.text_input("Enter Email")
            password = st.text_input("Enter Password", type="password")
            submitbtn = st.form_submit_button(label="Login")
            if submitbtn:
                user_file = pd.read_csv(USER_DATA_FILE)

                matched_user = user_file[user_file["Email"] == email]

                if not matched_user.empty:
                    saved_password = str(matched_user.iloc[0]["Pass"]).strip()
                    if password == saved_password:
                        st.session_state.authenticated = True
                        st.session_state.user_id = email
                        st.query_params.update(auth=True, user=email)
                        st.success("Login Successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password!")
                else:
                    st.error("Email does not exist! Please sign up first.")





















if st.session_state.authenticated:

    with st.sidebar:
        selected = option_menu(
            menu_title="mask detection",
            options= ["mask detect","LogOut"],
            icons=["🏠 ","📊", "🔮", "📈    ","🤖"],
            menu_icon="cast",
            default_index=0)

    if selected == "mask detect":
        st.title("😷 Real-Time Mask Detection App")

        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])

        # Load model
        model = load_model('C:/sem8 intern/cv/model/resnet152v2.h5')

        # Constants
        img_width, img_height = 224, 224
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Make sure folders exist
        import re

        user_email = st.session_state.user_id  # Gets logged-in user's email
        safe_email = re.sub(r'[^\w\-]', '_', user_email)  # Sanitize for filenames

        mask_dir = f'faces/{safe_email}/with_mask'
        nomask_dir = f'faces/{safe_email}/without_mask'
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(nomask_dir, exist_ok=True)

        # Start camera
        cap = cv2.VideoCapture(0)
        img_count = 0

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not found or failed to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 6)

            for (x, y, w, h) in faces:
                img_count += 1
                face_img = frame[y:y + h, x:x + w]

                # Preprocess
                resized_face = cv2.resize(face_img, (img_width, img_height))
                img_array = img_to_array(resized_face) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                pred_prob = model.predict(img_array)
                pred = np.argmax(pred_prob)

                if pred == 0:
                    label = "Mask"
                    color = (0, 255, 0)
                    save_path = f'{mask_dir}/{img_count}_mask.jpg'
                else:
                    label = "No Mask"
                    color = (0, 0, 255)
                    save_path = f'{nomask_dir}/{img_count}_nomask.jpg'

                cv2.imwrite(save_path, face_img)

                # Draw on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), font, 0.8, color, 2)

            # Convert to RGB for Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()

    if selected == "LogOut":
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.query_params.clear()
        st.success("Logged out Successfully.......Log In again to use out website")
        st.rerun()










