import streamlit as st
import cv2
import numpy as np
from fer import FER
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, ClientSettings
import av
import threading
import time  # Import time module for timing prompts

# Set page configuration
st.set_page_config(
    page_title="Real-Time Emotion Detection",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default hamburger menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .title {
        font-size: 2.5em;
        text-align: center;
        color: #4CAF50;
        padding: 20px 0;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">Real-Time Emotion Detection App 😊</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar.expander("⚙️ Settings", expanded=True):
    st.markdown("Configure the emotion detection parameters:")

    # Emotion Detection Options
    if "emotion_detection_enabled" not in st.session_state:
        st.session_state["emotion_detection_enabled"] = True

    if "show_emotion_stats" not in st.session_state:
        st.session_state["show_emotion_stats"] = False

    emotion_detection_enabled = st.checkbox(
        "Enable Emotion Detection",
        value=st.session_state["emotion_detection_enabled"],
    )
    st.session_state["emotion_detection_enabled"] = emotion_detection_enabled

    show_emotion_stats = st.checkbox(
        "Show Emotion Statistics", value=st.session_state["show_emotion_stats"]
    )
    st.session_state["show_emotion_stats"] = show_emotion_stats

    # Video Settings
    frame_rate = st.slider("Frame Rate", min_value=5, max_value=30, value=15, step=1)
    resolution = st.selectbox("Resolution", ["480p", "720p", "1080p"], index=1)

# Custom Client Settings for WebRTC
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 1920
                if resolution == "1080p"
                else 1280
                if resolution == "720p"
                else 640
            },
            "height": {
                "ideal": 1080
                if resolution == "1080p"
                else 720
                if resolution == "720p"
                else 480
            },
            "frameRate": {"ideal": frame_rate},
        },
        "audio": False,
    },
)


class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.emotion_counts = {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "sad": 0,
            "surprise": 0,
            "neutral": 0,
        }
        self.lock = threading.Lock()
        self.emotion_detection_enabled = True  # Default value
        self.current_emotion = None
        self.current_confidence = 0.0
        # Fun prompts with increased variety
        self.fun_prompts = {
            "happy": [
                "Keep smiling! 😊",
                "Your smile lights up the room!",
                "Happiness looks great on you!",
                "Your positivity is infectious!",
                "You brighten everyone's day!",
            ],
            "sad": [
                "Sending you a virtual hug! 🤗",
                "It's okay to feel sad sometimes.",
                "We're here for you!",
                "This too shall pass.",
                "Take your time to heal.",
            ],
            "angry": [
                "Take a deep breath. 🧘",
                "Let it out, it's okay.",
                "Remember to stay calm.",
                "Peace begins with a smile.",
                "Channel that energy positively!",
            ],
            "surprise": [
                "Wow! What's the surprise? 🎉",
                "Didn't see that coming!",
                "You look surprised!",
                "Life is full of surprises!",
                "Expect the unexpected!",
            ],
            "fear": [
                "It's okay to feel afraid.",
                "Courage is facing fear.",
                "You're stronger than you think!",
                "Fear is temporary, regret is forever.",
                "Embrace your fears and overcome them!",
            ],
            "disgust": [
                "Is something bothering you?",
                "Not a fan?",
                "Maybe try something else?",
                "Change can be good!",
                "Find what brings you joy!",
            ],
            "neutral": [
                "Hope you're having a good day!",
                "All good?",
                "Just chilling?",
                "Stay calm and carry on!",
                "Embrace the tranquility!",
            ],
        }
        self.current_prompt = ""
        self.last_prompt_time = time.time()
        self.prompt_interval = 5  # seconds

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.emotion_detection_enabled:
            # Detect emotions in the frame
            emotions = self.detector.detect_emotions(img)

            if emotions:
                for emotion in emotions:
                    (x, y, w, h) = emotion["box"]
                    # Get dominant emotion for the face
                    emotion_scores = emotion["emotions"]
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    score = emotion_scores[dominant_emotion]

                    # Update emotion counts with lock
                    with self.lock:
                        if dominant_emotion in self.emotion_counts:
                            self.emotion_counts[dominant_emotion] += 1
                        else:
                            self.emotion_counts[dominant_emotion] = 1
                        self.current_emotion = dominant_emotion
                        self.current_confidence = score

                    # Draw rectangle around the face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Put emotion text
                    cv2.putText(
                        img,
                        f"{dominant_emotion} ({score*100:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                # Update prompt if interval has passed
                if self.current_emotion and (
                    time.time() - self.last_prompt_time > self.prompt_interval
                ):
                    with self.lock:
                        prompts = self.fun_prompts.get(self.current_emotion, [])
                        if prompts:
                            self.current_prompt = np.random.choice(prompts)
                        else:
                            self.current_prompt = ""
                        self.last_prompt_time = time.time()
                # Draw the prompt on the image
                if self.current_prompt:
                    cv2.putText(
                        img,
                        self.current_prompt,
                        (10, 30),  # Position
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (30,144,255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                # If no face is detected, reset current_emotion and prompt
                with self.lock:
                    self.current_emotion = None
                    self.current_confidence = 0.0
                    self.current_prompt = ""

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    col1, col2 = st.columns([2, 1])

    with col1:
        # Initialize WebRTC streamer
        ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            client_settings=WEBRTC_CLIENT_SETTINGS,
            video_transformer_factory=EmotionDetector,
            async_processing=True,
        )

    with col2:
        if ctx.state.playing:
            if ctx.video_transformer:
                ctx.video_transformer.emotion_detection_enabled = (
                    st.session_state["emotion_detection_enabled"]
                )

                if st.session_state["show_emotion_stats"]:
                    with ctx.video_transformer.lock:
                        emotion_counts = ctx.video_transformer.emotion_counts.copy()
                    st.subheader("📊 Emotion Statistics")
                    st.bar_chart(emotion_counts)

                with ctx.video_transformer.lock:
                    current_emotion = ctx.video_transformer.current_emotion
                    current_confidence = ctx.video_transformer.current_confidence

                st.subheader("😊 Current Emotion")
                if current_emotion:
                    st.write(
                        f"{current_emotion.capitalize()} ({current_confidence*100:.1f}%)"
                    )
                else:
                    st.write("No face detected")
            else:
                st.write("Loading video processor...")
        else:
            st.write("Click on **Start** to begin emotion detection.")

    # Footer
    st.markdown(
        """
        <div class="footer">
            <hr>
            <p>Built by Angshuman Roy using Streamlit and FER</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
