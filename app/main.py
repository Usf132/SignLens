import sys
from pathlib import Path

# Make the project root importable regardless of the working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import cv2
import numpy as np
import os
from collections import deque
import time
import plotly.graph_objects as go
import toml
from backend import predict_asl_sign_lm, Text_to_speech

st.set_page_config(initial_sidebar_state="collapsed")

st.set_page_config(
    page_title="Sign Language Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


LOGO_PATH = ROOT_DIR / "assets" / "Logo.png"
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)


with st.sidebar:
    st.title("Sign Language to English")

    st.sidebar.divider()

    page = st.radio('Select Department:',
                   [" Live Translation", " Upload Video"," Text to Audio"])

    st.divider()

    # Model Status
    st.title("  Model Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", "95.02%", "Threshold: 70%")
    with col2:
        fps_placeholder = st.empty()
        fps_placeholder.metric("FPS", "0.0")

    # Confidence Threshold
    confidence = st.slider("Confidence Threshold", 0, 100, 55)
    stability_frames = st.slider("Stability Frames (Buffer)", 1, 30, 10, help="fps")

st.sidebar.divider()

st.sidebar.title("Customization:")

bg_color = st.sidebar.color_picker("Background color:", "#0a0e27")
sidebar_color = st.sidebar.color_picker("Sidebar Color:", "#0f0f23")
primary_color = st.sidebar.color_picker("primary color:", "#1f77b4")


if st.sidebar.button("Save changes"):
    config = {
        "theme": {
            "backgroundColor": bg_color,
            "secondaryBackgroundColor": sidebar_color,
            "primaryColor": primary_color,
            "textColor": "#ffffff"
        }
    }

    os.makedirs(".streamlit", exist_ok=True)
    with open(".streamlit/config.toml", "w") as f:
        toml.dump(config, f)

    st.sidebar.success("Done! Relode the page please!")


st.sidebar.divider()
st.sidebar.subheader("Voice Settings")


voice_presets = {
    "Man": "pNInz6obpgDQGcFmaJgB",
}

selected_voice_label = st.sidebar.selectbox("Select speaker's age/type:", options=list(voice_presets.keys()))
st.session_state.selected_voice_id = voice_presets[selected_voice_label]


if 'sequence' not in st.session_state:
    st.session_state.sequence = deque()
if 'conf_history' not in st.session_state:
    st.session_state.conf_history = deque(maxlen=50)
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False
if 'last_detected_letter' not in st.session_state:
    st.session_state.last_detected_letter = ""
if 'letter_counter' not in st.session_state:
    st.session_state.letter_counter = 0
if 'miss_streak' not in st.session_state:
    st.session_state.miss_streak = 0
if 'vid_sequence' not in st.session_state:
    st.session_state.vid_sequence = []
if 'last_video_frame' not in st.session_state:
    st.session_state.last_video_frame = None
if 'vid_frame_pos' not in st.session_state:
    st.session_state.vid_frame_pos = 0


if page == " Live Translation":
    box_html = """
    <div style="border: 2px solid #1ba1c2; padding: 15px; border-radius: 25px; text-align: center; background-color: transparent;">
        <h2 style="color: #1ba1c2; margin: 0;">Live Translation</h2>
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)
    st.header("Live Camera Feed & Translation")

    col_video, col_text = st.columns([2, 1])

    with col_text:
        st.subheader("Translation Output")
        text_placeholder = st.empty()

        st.subheader("Recognized Sequence")
        seq_placeholder = st.empty()

        btn_speak, btn_space, btn_del = st.columns([2, 1, 1])


        if btn_speak.button("Play", icon=":material/volume_up:", use_container_width=True, help="Play sound"):
            current_word = "".join(list(st.session_state.sequence)).title()
            if current_word:
                with st.spinner(f"Generating audio ({selected_voice_label})..."):
                    audio_data = Text_to_speech(current_word, st.session_state.selected_voice_id)
                    if audio_data:
                        st.success(f"Voice used: {selected_voice_label}")
                        st.audio(audio_data, format="audio/mp3", autoplay=True) 
                    else:
                        st.error("Error generating audio. Check API key.")
            else:
                st.warning("No word to speak yet!")


        if btn_space.button("",icon=":material/space_bar:", use_container_width=True, help="Space"):
            st.session_state.sequence.append(" ")


        if btn_del.button("⌫", use_container_width=True, help="Del"):
            if len(st.session_state.sequence) > 0:
                st.session_state.sequence.pop()

        st.divider()
        st.subheader("Real-time Confidence")
        graph_placeholder = st.empty()

    with col_video:
        video_placeholder = st.empty()

        # ctrl
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        if ctrl1.button(" Start", use_container_width=True):
            st.session_state.run_camera = True
        if ctrl2.button(" Stop", use_container_width=True):
            st.session_state.run_camera = False
        if ctrl3.button("Clear", use_container_width=True):
            st.session_state.sequence.clear()
            st.session_state.conf_history.clear()
            st.session_state.last_detected_letter = ""
            st.session_state.letter_counter = 0

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Couldn't access the camera. Make sure it's connected and not used by another app.")
            st.session_state.run_camera = False

        DETECTION_INTERVAL = 2

        prev_time = time.time()
        frame_index = 0
        last_annotated_frame = None
        last_letter = ""
        last_conf = 0.0

        while st.session_state.run_camera and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            elapsed = current_time - prev_time
            fps = 1 / elapsed if elapsed > 0 else 0.0
            prev_time = current_time

            fps_placeholder.metric("FPS", f"{fps:.1f}")

            run_detection = (frame_index % DETECTION_INTERVAL == 0)

            if run_detection:
              
  
                annotated_frame, letter, conf = predict_asl_sign_lm(frame, confidence)
                last_annotated_frame, last_letter, last_conf = annotated_frame, letter, conf
            else:
                
                annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
                letter, conf = last_letter, last_conf

            video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            text_placeholder.markdown(f"<div style='background-color: {sidebar_color}; padding: 30px; border-radius: 10px;'><h1 style='text-align: center; font-size: 80px; color: {primary_color}; margin:0;'>{letter}</h1></div>", unsafe_allow_html=True)

            MISS_TOLERANCE = 5  

            if run_detection:
                conf_percent = conf * 100
                st.session_state.conf_history.append(conf_percent)

                if conf_percent >= confidence:
                    st.session_state.miss_streak = 0

                    if letter == st.session_state.last_detected_letter:
                        st.session_state.letter_counter += 1
                    else:
                        st.session_state.last_detected_letter = letter
                        st.session_state.letter_counter = 1

                    if st.session_state.letter_counter == stability_frames:
                        if letter == "Delete":
                            if len(st.session_state.sequence) > 0:
                                st.session_state.sequence.pop()
                            st.session_state.letter_counter = 0
                            st.session_state.last_detected_letter = ""
                        elif letter == "Clear":
                            st.session_state.sequence.clear()
                            st.session_state.letter_counter = 0
                            st.session_state.last_detected_letter = ""
                        elif letter == "Space":
                            if len(st.session_state.sequence) == 0 or st.session_state.sequence[-1] != " ":
                                st.session_state.sequence.append(" ")
                            st.session_state.letter_counter = 0
                            st.session_state.last_detected_letter = ""
                        else:
                            if len(st.session_state.sequence) == 0 or st.session_state.sequence[-1] != letter:
                                st.session_state.sequence.append(letter)
                else:
                
                    st.session_state.miss_streak += 1
                    if st.session_state.miss_streak > MISS_TOLERANCE:
                        st.session_state.letter_counter = 0
                        st.session_state.last_detected_letter = ""
                        st.session_state.miss_streak = 0
            current_word = "".join(list(st.session_state.sequence))
            formatted_word = current_word.title()

            if formatted_word:
                seq_placeholder.info(f"**Word:** {formatted_word}")
            else:
                seq_placeholder.info("Waiting for signs...")

            if frame_index % 6 == 0:
                fig = go.Figure(data=go.Scatter(y=list(st.session_state.conf_history), mode='lines', line=dict(color=primary_color)))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(range=[0, 100]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"conf_chart_{frame_index}")

            frame_index += 1

        cap.release()


elif page == " Upload Video":
    box_html = """
        <div style="border: 2px solid #1ba1c2; padding: 15px; border-radius: 25px; text-align: center; background-color: transparent;">
            <h2 style="color: #1ba1c2; margin: 0;">Upload Video</h2>
        </div>
        """
    st.markdown(box_html, unsafe_allow_html=True)
    st.header("Upload Video for Translation")

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:

        TMP_DIR = ROOT_DIR / "tmp"
        TMP_DIR.mkdir(exist_ok=True)
        TEMP_VIDEO_PATH = TMP_DIR / "uploaded_video.mp4"

        with open(TEMP_VIDEO_PATH, "wb") as f:
            f.write(uploaded_file.read())

        st.success("Video Uploaded Successfully!")

        col_vid, col_txt = st.columns([2, 1])

        with col_txt:
            st.subheader("Translation Output")
            txt_place = st.empty()

            st.subheader("Recognized Sequence")
            seq_placeholder_vid = st.empty()
            
   
            current_vid_word = "".join(st.session_state.vid_sequence).title()
            if current_vid_word:
                seq_placeholder_vid.info(f"**Word:** {current_vid_word}")
            else:
                seq_placeholder_vid.info("Waiting for signs...")

      
            btn_speak_vid, btn_stop_vid, btn_resume_vid, btn_del_vid = st.columns([1.5, 1, 1, 1])

            if btn_speak_vid.button("Play", icon=":material/volume_up:", key="vid_speak", use_container_width=True, help="Play sound"):
                if current_vid_word:
                    with st.spinner(f"Generating audio ({selected_voice_label})..."):
                        audio_data = Text_to_speech(current_vid_word, st.session_state.selected_voice_id)
                        if audio_data:
                            st.success(f"Voice used: {selected_voice_label}")
                            st.audio(audio_data, format="audio/mp3", autoplay=True) 
                        else:
                            st.error("Error generating audio. Check API key.")
                else:
                    st.warning("No word to speak yet!")

            if btn_stop_vid.button("", icon=":material/pause:", key="vid_stop", use_container_width=True, help="Pause Video"):
                st.info("Video paused.")
                
            resume_clicked = btn_resume_vid.button("", icon=":material/play_circle:", key="vid_resume", use_container_width=True, help="Continue from where it stopped")

  
            if btn_del_vid.button("", icon=":material/backspace:", key="vid_del", use_container_width=True, help="Delete Letter"):
                if len(st.session_state.vid_sequence) > 0:
                    st.session_state.vid_sequence.pop()
                    st.rerun() 
         
        with col_vid:
            vid_place = st.empty()


            start_btn = st.button("Start Process", icon=":material/play_arrow:", use_container_width=True, help="Start from beginning")

           
            if start_btn or resume_clicked:
                cap = cv2.VideoCapture(str(TEMP_VIDEO_PATH))

                if not cap.isOpened():
                    st.error("Couldn't open the uploaded video file.")
                else:
                   
                    if start_btn:
                        st.session_state.vid_sequence.clear()
                        st.session_state.vid_frame_pos = 0
                    
               
                    elif resume_clicked:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.vid_frame_pos)

                    frame_counter = int(st.session_state.vid_frame_pos)
                    frame_skip = 2

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_counter += 1
                        st.session_state.vid_frame_pos = frame_counter 

                        if frame_counter % frame_skip != 0:
                            continue

                        annotated_frame, letter, conf = predict_asl_sign_lm(frame, confidence)

                        vid_place.image(annotated_frame, channels="BGR", use_container_width=True)

                        txt_place.markdown(f"<div style='background-color: {sidebar_color}; padding: 30px; border-radius: 10px;'><h1 style='text-align: center; font-size: 80px; color: {primary_color}; margin:0;'>{letter}</h1></div>", unsafe_allow_html=True)

                        if letter == "delete":
                            if len(st.session_state.vid_sequence) > 0:
                                st.session_state.vid_sequence.pop()
                        elif letter == "clear":
                            st.session_state.vid_sequence.clear()
                        elif letter == "space":
                            if len(st.session_state.vid_sequence) == 0 or st.session_state.vid_sequence[-1] != " ":
                                st.session_state.vid_sequence.append(" ")
                        elif letter and letter not in st.session_state.vid_sequence:
                            st.session_state.vid_sequence.append(letter)

                        seq_placeholder_vid.info(f"**Word:** {''.join(st.session_state.vid_sequence).title()}")
                        
                        st.session_state.last_video_frame = annotated_frame

                    cap.release()
                    st.success("Video processing complete!")
            
       
            elif st.session_state.last_video_frame is not None:
                vid_place.image(st.session_state.last_video_frame, channels="BGR", use_container_width=True)

elif page == " Text to Audio":
    box_html = """
        <div style="border: 2px solid #1ba1c2; padding: 15px; border-radius: 25px; text-align: center; background-color: transparent;">
            <h2 style="color: #1ba1c2; margin: 0;">Text to Audio</h2>
        </div>
        """
    st.markdown(box_html, unsafe_allow_html=True)
    st.header("Convert Text to Speech")

    st.write("Type any English text below and convert it to lifelike speech using the selected voice.")

   
    user_text = st.text_area("Enter your text here:", height=200, placeholder="Type something like: Hello, how are you today?")

  
    col_empty1, col_btn, col_empty2 = st.columns([1, 2, 1])

    with col_btn:
        if st.button("Generate & Play Audio", icon=":material/volume_up:", use_container_width=True):
            if user_text.strip():
                with st.spinner(f"Generating audio ({selected_voice_label})..."):
              
                    audio_data = Text_to_speech(user_text.strip(), st.session_state.selected_voice_id)
                    
                    if audio_data:
                        st.success("Audio generated successfully!")
                        st.audio(audio_data, format="audio/mp3", autoplay=True)
                    else:
                        st.error("Error generating audio. Check API key.")
            else:
                st.warning("Please enter some text first!")
