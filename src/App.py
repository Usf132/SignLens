import streamlit as st
import cv2
import numpy as np
from collections import deque
import time
import plotly.graph_objects as go
import toml
from Backend import predict_asl_sign,Text_to_speech

st.set_page_config(
    page_title="Sign Language Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
    
    with open(".streamlit/config.toml", "w") as f:
        toml.dump(config, f)
    
    st.sidebar.success("Done! Relode the page please!")

with st.sidebar:
    st.title("Sign Language to English")
    
    page = st.radio('Select Department:', 
                   ["🎥 Live Translation", "📹 Upload Video"])
    
    st.divider()
    
    # Model Status
    st.subheader("  Model Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", "92.4%", "Threshold: 70%")
    with col2:
        st.metric("FPS", "24.7")
    
    # Confidence Threshold
    confidence = st.slider("Confidence Threshold", 0, 100, 70)
    stability_frames = st.slider("Stability Frames (Buffer)", 1, 30, 10, help="fps")

st.sidebar.divider()
st.sidebar.subheader("🎙️ Voice Settings")


voice_presets = {
    "Man": "pNInz6obpgDQGcFmaJgB",
}

selected_voice_label = st.sidebar.selectbox("Select speaker's age/type:", options=list(voice_presets.keys()))
st.session_state.selected_voice_id = voice_presets[selected_voice_label]



if 'sequence' not in st.session_state:
    st.session_state.sequence = deque() #
if 'conf_history' not in st.session_state:
    st.session_state.conf_history = deque(maxlen=50) 
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False 
if 'last_detected_letter' not in st.session_state:
    st.session_state.last_detected_letter = ""
if 'letter_counter' not in st.session_state:
    st.session_state.letter_counter = 0


if page == "🎥 Live Translation":
    st.header("Live Camera Feed & Translation")
    
    col_video, col_text = st.columns([2, 1])
    
    with col_text:
        st.subheader("Translation Output")
        text_placeholder = st.empty()
        
        st.subheader("Recognized Sequence")
        seq_placeholder = st.empty()
        
        
        if st.button("Speak Word", width="stretch"):
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
        
        st.divider()
        st.subheader("Real-time Confidence")
        graph_placeholder = st.empty()

    with col_video:
        video_placeholder = st.empty()
        
        # ctrl
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        if ctrl1.button(" Start", width="stretch"):
            st.session_state.run_camera = True
        if ctrl2.button(" Stop", width="stretch"):
            st.session_state.run_camera = False
        if ctrl3.button ( "Clear", width="stretch"):
            st.session_state.sequence.clear()
            st.session_state.conf_history.clear()
            st.session_state.last_detected_letter = ""
            st.session_state.letter_counter = 0

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        
        while st.session_state.run_camera and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, letter, conf = predict_asl_sign(frame_rgb,confidence) 
            conf_percent = conf * 100
            
            st.session_state.conf_history.append(conf_percent)
            
            if conf_percent >= confidence:
                if letter == st.session_state.last_detected_letter:
                    st.session_state.letter_counter += 1
                else:
                    st.session_state.last_detected_letter = letter
                    st.session_state.letter_counter = 1
                
                if st.session_state.letter_counter == stability_frames:
                    if len(st.session_state.sequence) == 0 or st.session_state.sequence[-1] != letter:
                        st.session_state.sequence.append(letter)
            else:
                st.session_state.letter_counter = 0
            
            
            video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
            
            text_placeholder.markdown(f"<div style='background-color: {sidebar_color}; padding: 30px; border-radius: 10px;'><h1 style='text-align: center; font-size: 80px; color: {primary_color}; margin:0;'>{letter}</h1></div>", unsafe_allow_html=True)
            
            current_word = "".join(list(st.session_state.sequence))

            formatted_word = current_word.title()

            if formatted_word:
                seq_placeholder.info(f"**Word:** {formatted_word}")
            else:
                seq_placeholder.info("Waiting for signs...")
            
            
            fig = go.Figure(data=go.Scatter(y=list(st.session_state.conf_history), mode='lines', line=dict(color=primary_color)))
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(range=[0, 100]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"conf_chart_{time.time()}")
            time.sleep(0.01) 
            
        cap.release()



elif page == "📹 Upload Video":
    st.header("Upload Video for Translation 📂")
    
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
            
        st.success("Video Uploaded Successfully!")
        
        col_vid, col_txt = st.columns([2, 1])
        
        with col_txt:
            st.subheader("Translation Output")
            txt_place = st.empty()
            
            st.subheader("Recognized Sequence")
            seq_placeholder_vid = st.empty()
            
            # مكان عشان نعرض فيه مشغل الصوت بعد ما يخلص
            audio_placeholder = st.empty() 
            
        with col_vid:
            vid_place = st.empty()
            
            if st.button("Process Video", use_container_width=True):
                cap = cv2.VideoCapture("temp_video.mp4")
                
                current_word_vid = []
                frame_counter = 0 
                frame_skip = 2
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    frame_counter += 1

                    if frame_counter % frame_skip != 0:
                        continue
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    annotated_frame, letter, conf = predict_asl_sign(frame_rgb, confidence)
                    
                    vid_place.image(annotated_frame, channels="RGB", use_container_width=True)
                    
                    txt_place.markdown(f"<div style='background-color: {sidebar_color}; padding: 30px; border-radius: 10px;'><h1 style='text-align: center; font-size: 80px; color: {primary_color}; margin:0;'>{letter}</h1></div>", unsafe_allow_html=True)
                    
                    if letter and letter not in current_word_vid: 
                        current_word_vid.append(letter)
                        
                    seq_placeholder_vid.info(f"**Word:** {''.join(current_word_vid).title()}")
                    
                    #time.sleep(0.03)
                    
                cap.release()

                final_word = "".join(current_word_vid).title()
                if final_word:
                    with st.spinner(f"Generating audio ({selected_voice_label})..."):
                        audio_data = Text_to_speech(final_word, st.session_state.selected_voice_id)
                        
                        if audio_data:
                            st.success("Video processing complete!")
                            audio_placeholder.audio(audio_data, format="audio/mp3", autoplay=True)
                        else:
                            st.error("Error generating audio. Check API key.")