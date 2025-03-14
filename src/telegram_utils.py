import asyncio
import os

from dotenv import load_dotenv
from telegram import Bot

load_dotenv()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")


def send_telegram_notification(image_path, video_path, caption_dict=None):
    if caption_dict is None:
        caption_dict = {}

    async def _send_notification():
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        # Send snapshot image
        try:
            with open(image_path, "rb") as image_file:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=image_file,
                    caption=caption_dict.get("image_caption", ""),
                )
            print(f"Sent snapshot '{image_path}' successfully.")
        except Exception as e:
            print("Error sending snapshot:", e)
        # Send video clip
        try:
            with open(video_path, "rb") as video_file:
                await bot.send_video(
                    chat_id=TELEGRAM_CHAT_ID,
                    video=video_file,
                    caption=caption_dict.get("video_caption", ""),
                )
            print(f"Sent video '{video_path}' successfully.")
        except Exception as e:
            print("Error sending video:", e)

    asyncio.run(_send_notification())
