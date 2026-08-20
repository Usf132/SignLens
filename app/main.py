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
import queue
import av
import plotly.graph_objects as go
import toml
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from backend import predict_asl_sign_lm, Text_to_speech

# FIX: st.set_page_config() can only be called once per app -- the original
# file called it twice, which raises StreamlitAPIException and crashes the
# app on every run. Kept only the fuller call below.
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

    # FIX: was a relative ".streamlit" path, which depends on the process's
    # working directory and can silently write to the wrong place. Anchor it
    # to ROOT_DIR like everything else in this file already does.
    streamlit_dir = ROOT_DIR / ".streamlit"
    streamlit_dir.mkdir(exist_ok=True)
    with open(streamlit_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    # FIX: typo "Relode" -> "Reload"
    st.sidebar.success("Done! Reload the page please!")


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
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=1)
if 'vid_paused' not in st.session_state:
    st.session_state.vid_paused = False


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
        # NOTE: no more manual video_placeholder.image() -- webrtc_streamer
        # renders the live (annotated) video itself, directly in the browser.

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
        # FIX: cv2.VideoCapture(0) opens a camera attached to the SERVER,
        # which doesn't exist on Streamlit Community Cloud (or any hosted
        # environment) -- that's the "can't open camera by index" error.
        # streamlit-webrtc instead streams frames from the VISITOR'S
        # browser webcam over WebRTC to this backend for processing.

        DETECTION_INTERVAL = 2
        # Mutable cell so the callback closure (runs in a background thread
        # managed by streamlit-webrtc) can keep a running frame count.
        frame_counter = {"n": 0}

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            frame_counter["n"] += 1

            if frame_counter["n"] % DETECTION_INTERVAL == 0:
                annotated_frame, letter, conf = predict_asl_sign_lm(img, confidence)
                # IMPORTANT: this callback runs in a background thread, so it
                # must NOT touch st.session_state or any Streamlit widget
                # directly -- only hand results off through a thread-safe
                # queue. The polling loop below (running in Streamlit's
                # normal script thread) does all session_state / UI updates.
                try:
                    st.session_state.result_queue.put_nowait({"letter": letter, "confidence": conf})
                except queue.Full:
                    pass
            else:
                annotated_frame = img

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        # FIX: Streamlit Community Cloud blocks the outbound UDP that plain
        # STUN needs to negotiate a connection -- that's what was producing
        # the repeating "aioice ... AttributeError: 'NoneType' object has no
        # attribute 'sendto'" errors in the logs (ICE candidate gathering
        # timing out, then a stale retry firing after teardown). A TURN
        # server relays media over TCP/TLS instead, which isn't blocked.
        # get_ice_servers() below tries Twilio's free dynamic TURN
        # credentials first (needs a Twilio account, see note below), and
        # falls back to Open Relay Project's free public TURN server if
        # Twilio isn't configured -- good enough to get you working today,
        # but swap in your own TURN provider for anything beyond testing,
        # since free public TURN servers have no uptime guarantee.
        def get_ice_servers():
            try:
                from twilio.rest import Client
                account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
                auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
                client = Client(account_sid, auth_token)
                token = client.tokens.create()
                return token.ice_servers
            except (KeyError, FileNotFoundError, Exception):
                # No Twilio secrets configured (or the call failed) -- fall
                # back to Open Relay Project's free public TURN server.
                return [
                    {"urls": "stun:stun.relay.metered.ca:80"},
                    {
                        "urls": "turn:global.relay.metered.ca:80",
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                    {
                        "urls": "turn:global.relay.metered.ca:443",
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                    {
                        "urls": "turn:global.relay.metered.ca:443?transport=tcp",
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                ]

        webrtc_ctx = webrtc_streamer(
            key="sign-language-live",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": get_ice_servers()},
        )

        MISS_TOLERANCE = 5
        poll_count = 0

        while webrtc_ctx.state.playing and st.session_state.run_camera:
            try:
                result = st.session_state.result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            letter, conf = result["letter"], result["confidence"]
            poll_count += 1

            text_placeholder.markdown(
                f"<div style='background-color: {sidebar_color}; padding: 30px; border-radius: 10px;'>"
                f"<h1 style='text-align: center; font-size: 80px; color: {primary_color}; margin:0;'>{letter}</h1></div>",
                unsafe_allow_html=True,
            )

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

            if poll_count % 6 == 0:
                fig = go.Figure(data=go.Scatter(y=list(st.session_state.conf_history), mode='lines', line=dict(color=primary_color)))
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), yaxis=dict(range=[0, 100]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                graph_placeholder.plotly_chart(fig, use_container_width=True, key=f"conf_chart_{poll_count}")


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

            # FIX: this button used to only show a message -- it never
            # actually stopped the frame loop below, because nothing inside
            # that loop checked any pause flag. Now it sets vid_paused and
            # reruns, and the while-loop checks it every iteration.
            if btn_stop_vid.button("", icon=":material/pause:", key="vid_stop", use_container_width=True, help="Pause Video"):
                st.session_state.vid_paused = True
                st.info("Video paused.")
                st.rerun()

            resume_clicked = btn_resume_vid.button("", icon=":material/play_circle:", key="vid_resume", use_container_width=True, help="Continue from where it stopped")
            if resume_clicked:
                st.session_state.vid_paused = False

  
            if btn_del_vid.button("", icon=":material/backspace:", key="vid_del", use_container_width=True, help="Delete Letter"):
                if len(st.session_state.vid_sequence) > 0:
                    st.session_state.vid_sequence.pop()
                    st.rerun() 
         
        with col_vid:
            vid_place = st.empty()


            start_btn = st.button("Start Process", icon=":material/play_arrow:", use_container_width=True, help="Start from beginning")

           
            if start_btn or resume_clicked:
                cap = cv2.VideoCapture(str(TEMP_VIDEO_PATH))
                # NOTE: unlike Live Translation, this VideoCapture reads an
                # uploaded FILE from disk, not a live camera device -- that
                # works fine on a server, no webrtc needed here.

                if not cap.isOpened():
                    st.error("Couldn't open the uploaded video file.")
                else:
                   
                    if start_btn:
                        st.session_state.vid_sequence.clear()
                        st.session_state.vid_frame_pos = 0
                        st.session_state.vid_paused = False
                    
               
                    elif resume_clicked:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.vid_frame_pos)

                    frame_counter = int(st.session_state.vid_frame_pos)
                    frame_skip = 2

                    while cap.isOpened():
                        # FIX: this is what actually makes the Pause button work --
                        # break out of the loop (and stop reading/processing more
                        # frames) as soon as vid_paused is set.
                        if st.session_state.vid_paused:
                            break

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

                        # FIX: labels from the model are capitalized
                        # ("Delete"/"Clear"/"Space" -- see models/class_mapping.json),
                        # but this branch compared against lowercase strings, so
                        # these commands never fired here (they only worked in
                        # Live Translation, which already used the right casing).
                        if letter == "Delete":
                            if len(st.session_state.vid_sequence) > 0:
                                st.session_state.vid_sequence.pop()
                        elif letter == "Clear":
                            st.session_state.vid_sequence.clear()
                        elif letter == "Space":
                            if len(st.session_state.vid_sequence) == 0 or st.session_state.vid_sequence[-1] != " ":
                                st.session_state.vid_sequence.append(" ")
                        # FIX: "letter not in st.session_state.vid_sequence" checked
                        # the ENTIRE sequence history, so a repeated letter (e.g. the
                        # second "L" in "HELLO") could never be added again once it
                        # had appeared anywhere. Now it only checks the last letter,
                        # matching the (already-correct) Live Translation logic.
                        elif letter and (len(st.session_state.vid_sequence) == 0 or st.session_state.vid_sequence[-1] != letter):
                            st.session_state.vid_sequence.append(letter)

                        seq_placeholder_vid.info(f"**Word:** {''.join(st.session_state.vid_sequence).title()}")
                        
                        st.session_state.last_video_frame = annotated_frame

                    cap.release()
                    if not st.session_state.vid_paused:
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
