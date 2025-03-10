import os
import cv2
import threading
import time
import subprocess
import wave
import random
from collections import deque
import logging

import pyaudio
from telegram_utils import send_telegram_notification

logger = logging.getLogger(__name__)
# ---------------------------
# Video Capture with Rolling Buffer
# ---------------------------
class VideoCaptureBuffer:
    def __init__(self, buffer_seconds=3, source=0):
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            logger.info("Warning: Unable to open video capture device!")
            self.video_available = False
        else:
            self.video_available = True
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30  # default FPS if not available
        self.buffer_seconds = buffer_seconds
        self.buffer_maxlen = int(buffer_seconds * self.fps)
        self.buffer = deque(maxlen=self.buffer_maxlen)
        self.running = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        if self.video_available:
            self.running = True
            self.thread.start()

    def stop(self):
        self.running = False
        if self.video_available:
            self.thread.join()
            self.capture.release()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            timestamp = time.time()
            with self.lock:
                self.buffer.append((timestamp, frame))
            # Sleep to match the capture FPS
            time.sleep(1.0 / self.fps)

    def get_video_clip_between(self, start_time, end_time):
        with self.lock:
            clip = [frame for (t, frame) in self.buffer if start_time <= t <= end_time]
        return clip

    def get_latest_frame(self):
        with self.lock:
            if self.buffer:
                return self.buffer[-1][1]
            return None


# ---------------------------
# Audio Capture with Rolling Buffer
# ---------------------------
class AudioCaptureBuffer:
    def __init__(self, buffer_seconds=6, rate=44100, channels=1, chunk=1024):
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16  # 16-bit audio
        self.audio_interface = pyaudio.PyAudio()
        self.buffer_maxlen = int((buffer_seconds * rate) / chunk)
        self.buffer = deque(maxlen=self.buffer_maxlen)
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stream = None
        self.audio_available = False

    def start(self):
        self.running = True
        try:
            self.stream = self.audio_interface.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.audio_available = True
            self.thread.start()
            logger.info("Audio stream started successfully.")
        except Exception as e:
            logger.info("Failed to start audio stream:", e)
            self.audio_available = False

    def stop(self):
        self.running = False
        if self.audio_available:
            self.thread.join()
            self.stream.stop_stream()
            self.stream.close()
        self.audio_interface.terminate()

    def _capture_loop(self):
        while self.running and self.audio_available:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                timestamp = time.time()
                self.buffer.append((timestamp, data))
            except Exception as e:
                logger.info("Error capturing audio:", e)
                break

    def get_audio_clip_between(self, start_time, end_time):
        clip = [data for (t, data) in self.buffer if start_time <= t <= end_time]
        return b''.join(clip)


# ---------------------------
# Saving Video and Audio Clips
# ---------------------------
def save_video_clip(frames, output_file, fps=30):
    if not frames:
        logger.info("No frames to save!")
        return False
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return True

def save_audio_clip(audio_data, output_file, rate=44100, channels=1, sample_width=2):
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(audio_data)
    wf.close()


# ---------------------------
# YOLO Inference Simulation
# ---------------------------
def detector_inference(frame, confidence_threshold=0.5):
    # Placeholder for actual YOLO detection
    pass
def simulate_yolo_detection(frame, confidence_threshold=0.5):
    """
    Simulate a YOLO detection.
    Replace this with your actual YOLO model inference using PyTorch.
    For now, we randomly trigger detection with a low probability.
    """
    if frame is None:
        return False, None
    detection_probability = 0.01  # simulate a 1% chance of detection per frame
    if random.random() < detection_probability:
        confidence = random.uniform(confidence_threshold, 1.0)
        detection_info = {"confidence": confidence, "label": "target_object"}
        return True, detection_info
    return False, None


# ---------------------------
# Event Processing and Notification
# ---------------------------
def process_event(event_time, video_buffer, audio_buffer):
    logger.info("Processing event at:", event_time)
    # Retrieve pre-event video from buffer
    pre_event_video = video_buffer.get_video_clip_between(event_time - 3, event_time)
    # Wait for post-event video to be captured
    time.sleep(3)
    post_event_video = video_buffer.get_video_clip_between(event_time, event_time + 3)
    full_video = pre_event_video + post_event_video

    # Save video clip
    video_temp_file = "event_video.mp4"
    if full_video and save_video_clip(full_video, video_temp_file, fps=video_buffer.fps):
        final_output_file = "event_clip.mp4"
        # Retrieve audio clip if available
        if audio_buffer.audio_available:
            full_audio = audio_buffer.get_audio_clip_between(event_time - 3, event_time + 3)
        else:
            full_audio = b''
            logger.info("No audio data captured.")

        if full_audio:
            audio_temp_file = "event_audio.wav"
            save_audio_clip(full_audio, audio_temp_file,
                            rate=audio_buffer.rate,
                            channels=audio_buffer.channels,
                            sample_width=2)
            # merge_command = [
            #     "ffmpeg", "-y",
            #     "-i", video_temp_file,
            #     "-i", audio_temp_file,
            #     "-c:v", "copy",
            #     "-c:a", "aac",
            #     final_output_file
            # ]
            # logger.info("Merging video and audio...")
            # subprocess.run(merge_command, check=True)
            # os.remove(audio_temp_file)
            # os.remove(video_temp_file)
            # logger.info(f"Saved event clip to '{final_output_file}'.")
        else:
            os.rename(video_temp_file, final_output_file)
            logger.info(f"Saved event clip (video only) to '{final_output_file}'.")
        # Save a snapshot from the middle of the pre-event clip (if available)
        if pre_event_video:
            snapshot = pre_event_video[len(pre_event_video)//2]
            snapshot_file = "snapshot.jpg"
            cv2.imwrite(snapshot_file, snapshot)
            logger.info(f"Snapshot saved to '{snapshot_file}'.")
            # Trigger Telegram notification with the snapshot and event video clip
            send_telegram_notification(snapshot_file, video_temp_file)
    else:
        logger.info("Failed to save video clip.")


# ---------------------------
# Detection Engine Thread
# ---------------------------
def detection_loop(video_buffer, audio_buffer, stop_event):
    last_event_time = 0
    cooldown_period = 5  # seconds between events to avoid duplicates
    while not stop_event.is_set():
        frame = video_buffer.get_latest_frame()
        if frame is not None:
            detected, detection_info = simulate_yolo_detection(frame)
            if detected:
                current_time = time.time()
                # Check cooldown to prevent multiple triggers from the same event
                if current_time - last_event_time > cooldown_period:
                    logger.info("Detection triggered with info:", detection_info)
                    last_event_time = current_time
                    # Process the event in a separate thread
                    event_thread = threading.Thread(target=process_event,
                                                    args=(current_time, video_buffer, audio_buffer),
                                                    daemon=True)
                    event_thread.start()
        time.sleep(0.1)  # check every 100ms


# ---------------------------
# Main: Capture, Detection, and Event Processing
# ---------------------------
if __name__ == "__main__":
    # Initialize video and audio buffers
    video_buffer = VideoCaptureBuffer(buffer_seconds=3)
    audio_buffer = AudioCaptureBuffer(buffer_seconds=6)

    video_buffer.start()
    audio_buffer.start()

    stop_event = threading.Event()
    detection_thread = threading.Thread(target=detection_loop,
                                        args=(video_buffer, audio_buffer, stop_event),
                                        daemon=True)
    detection_thread.start()

    try:
        logger.info("System running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping system...")
        stop_event.set()
    finally:
        video_buffer.stop()
        audio_buffer.stop()
        detection_thread.join()
