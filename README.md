# SafeHouse

A robust home security system powered by computer vision and machine learning models.

## Features

- **Real-Time Detection:** Uses YOLO (You Only Look Once) or SAM 2 (Segment Anything) models to identify people and objects
- **Smart Buffer Recording:** Captures video from before and after detection events
- **Instant Notifications:** Sends alerts with annotated images via Telegram when intruders are detected
- **Efficient Processing:** Analyzes only every 15th frame to reduce CPU load while maintaining security
- **Audio Recording:** Optional audio capture for comprehensive security monitoring
- **Low Resource Usage:** Designed to run on modest hardware

## Installation

```bash
# Clone the repository
git clone https://github.com/dnlgsv/safe_house.git
cd safe_house

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your Telegram bot
```

## Usage

```bash
# Run the application
python src/app.py
```

The system will start monitoring your camera feed. When an unknown person is detected, it will:

- Save a video clip containing footage from before and after the detection
- Send an annotated image and video clip to your Telegram account
- Continue monitoring for new events

## Configuration
