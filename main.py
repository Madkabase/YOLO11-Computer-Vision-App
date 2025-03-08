#Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
import time
from ultralytics import YOLO
from PIL import Image

#Get the absolute path of the current file
FILE = Path(__file__).resolve()

#Get the parent directory of the current file
ROOT = FILE.parent

#Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

#Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

#Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR/'video1.mp4',
    'video 2': VIDEO_DIR/'video2.mp4'
}

#Webcam Config
WEBCAM_DEFAULT_ID = 0  # Default webcam (usually the built-in webcam)

#Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'

#In case of your custom model
#DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt'

SEGMENTATION_MODEL  = MODEL_DIR/'yolo11n-seg.pt'

POSE_ESTIMATION_MODEL = MODEL_DIR/'yolo11n-pose.pt'

#Page Layout
st.set_page_config(
    page_title = "YOLO11",
    page_icon = "🤖"
)

#Header
st.header("Object Detection using YOLO11")

#SideBar
st.sidebar.header("Model Configurations")

#Choose Model: Detection, Segmentation or Pose Estimation
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])

#Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

#Selecting Detection, Segmentation, Pose Estimation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)
elif model_type ==  'Pose Estimation':
    model_path = Path(POSE_ESTIMATION_MODEL)

#Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified path: {model_path}")
    st.error(e)

#Image / Video / Webcam Configuration
st.sidebar.header("Input Configuration")
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST
)

source_image = None
if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, caption = "Default Image", use_container_width=True)
            else:
                uploaded_image  =Image.open(source_image)
                st.image(source_image, caption = "Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption = "Detected Image", use_container_width=True)
            else:
                if st.sidebar.button("Detect Objects"):
                    result = model.predict(uploaded_image, conf = confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:,:,::-1]
                    st.image(result_plotted, caption = "Detected Image", use_container_width=True)

                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)

elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a Video...", VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button("Detect Video Objects"):
            try:
                video_cap = cv2.VideoCapture(
                    str(VIDEOS_DICT.get(source_video))
                )
                st_frame = st.empty()
                while (video_cap.isOpened()):
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        #Predict the objects in the image using YOLO11
                        result = model.predict(image, conf = confidence_value)
                        #Plot the detected objects on the video frame
                        result_plotted = result[0].plot()
                        st_frame.image(result_plotted, caption = "Detected Video",
                                       channels = "BGR",
                                       use_container_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Loading Video"+str(e))

elif source_radio == WEBCAM:
    # Webcam settings
    webcam_id = st.sidebar.number_input(
        "Select Webcam ID", min_value=0, max_value=10, value=WEBCAM_DEFAULT_ID, step=1,
        help="Usually 0 is the built-in webcam, 1+ are external webcams"
    )
    
    fps_options = [5, 10, 15, 20, 25, 30]
    webcam_fps = st.sidebar.selectbox(
        "Frames Per Second", fps_options, index=1,
        help="Higher FPS means smoother video but more processing power required"
    )
    
    # Create a frame placeholder
    frame_placeholder = st.empty()
    
    # Add a stop button
    stop_button_col1, stop_button_col2 = st.columns([1, 5])
    with stop_button_col1:
        stop_button = st.button("Stop Webcam", key="stop_button")
    
    if st.sidebar.button("Start Webcam Detection"):
        try:
            # Open the webcam
            video_cap = cv2.VideoCapture(int(webcam_id))
            
            if not video_cap.isOpened():
                st.error(f"Could not open webcam with ID {webcam_id}. Please check your webcam connection and ID.")
            else:
                st.success(f"Webcam {webcam_id} opened successfully! Press 'Stop Webcam' to end.")
                
                # Counter for FPS calculation
                frame_count = 0
                start_time = time.time()
                
                # Continue until the stop button is pressed
                while not stop_button:
                    # Capture frame-by-frame
                    ret, frame = video_cap.read()
                    
                    # If frame read correctly
                    if ret:
                        # Resize frame to 16:9 aspect ratio
                        frame = cv2.resize(frame, (720, int(720 * (9/16))))
                        
                        # Predict objects in the frame
                        result = model.predict(frame, conf=confidence_value)
                        
                        # Plot the detected objects on the frame
                        result_plotted = result[0].plot()
                        
                        # Display the resulting frame
                        frame_placeholder.image(result_plotted, channels="BGR", 
                                               caption=f"Webcam Feed - {model_type}", 
                                               use_container_width=True)
                        
                        # Control frame rate for smoother display and to reduce CPU usage
                        frame_count += 1
                        elapsed_time = time.time() - start_time
                        
                        # Calculate and display FPS every second
                        if elapsed_time > 1.0:
                            fps = frame_count / elapsed_time
                            frame_count = 0
                            start_time = time.time()
                        
                        # Control the frame rate
                        time.sleep(max(0, 1.0/webcam_fps - (time.time() - start_time)))
                        
                        # Rerun to check the stop button status
                        if stop_button:
                            break
                    else:
                        st.error("Failed to capture frame from webcam. The webcam might be in use by another application.")
                        break
                        
                # Release the webcam when done
                video_cap.release()
                st.success("Webcam released successfully!")
                
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
            
            # Provide troubleshooting tips
            st.warning("""
            Troubleshooting tips:
            1. Make sure your webcam is connected and not being used by another application
            2. Try a different webcam ID if you have multiple cameras
            3. Restart your computer and try again
            4. Check if your browser has permission to access the webcam
            """)
    
    # Display instructions
    with st.expander("Webcam Instructions"):
        st.write("""
        ### How to use webcam detection:
        1. Select your webcam ID (usually 0 for built-in webcams)
        2. Choose the desired frames per second (FPS)
        3. Click 'Start Webcam Detection' to begin
        4. Click 'Stop Webcam' to end the detection
        
        ### Troubleshooting:
        - If you see an error, make sure your webcam is not being used by another application
        - Try different webcam IDs if you have multiple cameras
        - Reduce the FPS if the detection is too slow
        """)