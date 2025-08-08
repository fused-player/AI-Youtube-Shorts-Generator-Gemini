import cv2
import numpy as np
import webrtcvad
import wave
import contextlib
from pydub import AudioSegment
import os

# Update paths to the model files
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
temp_audio_path = "temp_audio.wav"

# Load DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3

def voice_activity_detection(audio_frame, sample_rate=16000):
    return vad.is_speech(audio_frame, sample_rate)

def extract_audio_from_video(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

global Frames
Frames = []  # [x,y,w,h]

def detect_faces_and_speakers(input_video_path, output_video_path):
    # Return Frames:
    global Frames
    Frames = []  # Reset frames list
    
    # Extract audio from the video
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Read the extracted audio
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    frame_duration_ms = 30  # 30ms frames
    audio_generator = process_audio_frame(audio_data, sample_rate, frame_duration_ms)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
        is_speaking_audio = voice_activity_detection(audio_frame, sample_rate)
        
        # Initialize variables with default values
        x, y, x1, y1 = 0, 0, w, h  # Default to full frame if no face detected
        MaxDif = 0
        Add = []
        
        # First pass: find maximum lip distance
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (temp_x, temp_y, temp_x1, temp_y1) = box.astype("int")
                face_width = temp_x1 - temp_x
                face_height = temp_y1 - temp_y

                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((temp_y + 2 * face_height // 3) - (temp_y1))
                Add.append([[temp_x, temp_y, temp_x1, temp_y1], lip_distance])

                MaxDif = max(lip_distance, MaxDif)  # Fixed: was == instead of =
        
        # Second pass: find the active speaker
        active_speaker_found = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (temp_x, temp_y, temp_x1, temp_y1) = box.astype("int")
                face_width = temp_x1 - temp_x
                face_height = temp_y1 - temp_y

                # Draw bounding box
                cv2.rectangle(frame, (temp_x, temp_y), (temp_x1, temp_y1), (0, 255, 0), 2)

                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((temp_y + 2 * face_height // 3) - (temp_y1))
                print(lip_distance)

                # Combine visual and audio cues
                if lip_distance >= MaxDif and is_speaking_audio:  # Adjust the threshold as needed
                    cv2.putText(frame, "Active Speaker", (temp_x, temp_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if lip_distance >= MaxDif:
                    # This is the active speaker, store their coordinates
                    x, y, x1, y1 = temp_x, temp_y, temp_x1, temp_y1
                    active_speaker_found = True
                    break

        # If no active speaker found but faces detected, use the first face
        if not active_speaker_found and len(Add) > 0:
            x, y, x1, y1 = Add[0][0]

        # Append frame data (always executed now)
        Frames.append([x, y, x1, y1])

        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Clean up temp audio file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)


if __name__ == "__main__":
    detect_faces_and_speakers("input_video.mp4", "output_video.mp4")
    print(Frames)
    print(len(Frames))
    if len(Frames) > 5:
        print(Frames[1:5])