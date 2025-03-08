import cv2
import threading
import time
import subprocess
import wave
import os
from collections import deque

import pyaudio

# ---------------------------
# Video Capture with Rolling Buffer
# ---------------------------
class VideoCaptureBuffer:
    def __init__(self, buffer_seconds=3, source=0):
        self.capture = cv2.VideoCapture(source)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        self.buffer_seconds = buffer_seconds
        self.buffer_maxlen = int(buffer_seconds * self.fps)
        self.buffer = deque(maxlen=self.buffer_maxlen)
        self.running = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
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

    def start(self):
        self.running = True
        self.stream = self.audio_interface.open(format=self.format,
                                                  channels=self.channels,
                                                  rate=self.rate,
                                                  input=True,
                                                  frames_per_buffer=self.chunk)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()

    def _capture_loop(self):
        while self.running:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            timestamp = time.time()
            self.buffer.append((timestamp, data))

    def get_audio_clip_between(self, start_time, end_time):
        clip = [data for (t, data) in self.buffer if start_time <= t <= end_time]
        return b''.join(clip)

def save_video_clip(frames, output_file, fps=30):
    if not frames:
        print("No frames to save!")
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def save_audio_clip(audio_data, output_file, rate=44100, channels=1, sample_width=2):
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(audio_data)
    wf.close()

# ---------------------------
# Main: Capture and Merge Video & Audio on Event
# ---------------------------
if __name__ == "__main__":
    # Video buffer holds 3 seconds (pre or post, as needed)
    video_buffer = VideoCaptureBuffer(buffer_seconds=3)
    # Audio buffer holds 6 seconds to cover 3 sec pre-event and 3 sec post-event
    audio_buffer = AudioCaptureBuffer(buffer_seconds=6)

    video_buffer.start()
    audio_buffer.start()

    try:
        print("Capturing video and audio... Waiting for event detection simulation.")
        # Let buffers fill
        time.sleep(5)
        
        # Simulate event detection
        event_time = time.time()
        print("Event detected at:", event_time)
        
        # Immediately capture pre-event video (last 3 seconds)
        pre_event_video = video_buffer.get_video_clip_between(event_time - 3, event_time)
        
        # Wait 3 seconds for post-event data to accumulate
        time.sleep(3)
        post_event_video = video_buffer.get_video_clip_between(event_time, event_time + 3)
        
        # Combine video frames
        full_video = pre_event_video + post_event_video
        
        # Get audio from 3 seconds before to 3 seconds after the event
        full_audio = audio_buffer.get_audio_clip_between(event_time - 3, event_time + 3)
        
        # Save temporary files
        video_temp_file = "event_video.mp4"
        audio_temp_file = "event_audio.wav"
        final_output_file = "event_clip.mp4"
        
        save_video_clip(full_video, video_temp_file, fps=video_buffer.fps)
        save_audio_clip(full_audio, audio_temp_file, rate=audio_buffer.rate, channels=audio_buffer.channels, sample_width=2)
        
        # Merge video and audio using FFmpeg
        merge_command = [
            "ffmpeg", "-y",
            "-i", video_temp_file,
            "-i", audio_temp_file,
            "-c:v", "copy",
            "-c:a", "aac",
            final_output_file
        ]
        print("Merging video and audio...")
        subprocess.run(merge_command, check=True)
        print(f"Saved event clip with audio to '{final_output_file}'.")
        
        # Clean up temporary files
        os.remove(video_temp_file)
        os.remove(audio_temp_file)
    
    except KeyboardInterrupt:
        print("Stopping capture.")
    finally:
        video_buffer.stop()
        audio_buffer.stop()
