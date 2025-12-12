# youtube_watcher_service.py
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import time
import re
from recommender_data_service import process_video

# OpenAI imports
from openai import OpenAI  

load_dotenv()

# --- CONFIGURATION ---
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (YOUTUBE_API_KEY and MONGO_URI and OPENAI_API_KEY):
    raise RuntimeError("Missing one of YOUTUBE_API_KEY, MONGO_CONNECTION_STRING, or OPENAI_API_KEY in env")

# --- CHANNELS TO MONITOR ---
CHANNELS = {
    "Babloo Lahori": "UCUMnLDbOryIo-gwmrLFo2qA",
    "ReviewsPK": "UCs2CReSOxze9hUknRowMdAA",
    "VideoWaliSarkar": "UCheoCqHDwPcfS9Jrgz8siQw",
    "MAS TECH": "UC_k-Bk8mErWg6kchpkw6Asg"
}

# --- DATABASE SETUP ---
client = MongoClient(MONGO_URI)
db = client["MobileDB"]
videos_collection = db["videos"]

# --- YOUTUBE SERVICE ---
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# --- OPENAI CLIENT SETUP ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def fetch_new_videos(channel_id):
    """Fetch latest videos from a channel in the last 7 days."""
    published_after = (datetime.utcnow() - timedelta(days=30)).isoformat("T") + "Z"
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        publishedAfter=published_after,
        type="video",
        maxResults=10
    )
    response = request.execute()
    return response.get("items", [])

def llm_relevance_check(title: str, description: str = "") -> bool:
    """
    Use an OpenAI model to decide whether a video is a list/comparison video (multiple phones / price-range).
    Returns True if relevant (YES), False otherwise (NO).
    """
    # Compose the instruction/prompt succinctly and deterministically.
    system_instruction = (
        """You are a labeler that answers only YES or NO.
Return exactly YES or NO (uppercase), nothing else.

Answer YES if the video is very likely a list-style recommendation video of smartphones, even if the title does not explicitly mention the number of phones. This includes:
- Category-based lists (best camera phones, best gaming phones, best performance phones, best battery phones, etc.)
- Titles with phrases like “My Choices”, “Top Picks”, “Best Smartphones”, “Best Phones for 2024/2025”, etc.
- Videos that normally include multiple phones even when the count is not stated.

Assume that any “Best ___ Phones” video typically includes 4 or more phones unless the title clearly indicates otherwise.

Answer NO if:
- The video reviews a single phone.
- The video compares only 2 or 3 phones (A vs B, A vs B vs C, “Top 3 phones”, etc.).
- The video is not about listing multiple recommended phones.
- The video is about news, leaks, rumors, or feature explanations without listing multiple phones.

If the title suggests a list but the number of phones is unclear, lean toward YES unless it explicitly states 2 or 3 phones.
"""
    )

    user_input = f"Title: {title}\n\nDescription: {description}"

    try:
        resp = openai_client.responses.create(
            model="gpt-4o",          # adjust if you want another model
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input}
            ],
            max_output_tokens=20
        )
        # The Responses API returns text in various places depending on version;
        # prefer top-level output_text if present, else join output segments.
        out_text = ""
        if hasattr(resp, "output_text") and resp.output_text:
            out_text = resp.output_text
        else:
            # fallback: try to join choices / content segments
            segments = []
            for item in getattr(resp, "output", []) or []:
                # item may have 'content' list of dicts
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        segments.append(c.get("text", ""))
            out_text = " ".join(segments)

        decision = out_text.strip().upper()
        # Accept "YES" if it appears at start to be lenient with trailing punctuation/newlines
        return decision.startswith("YES")
    except Exception as e:
        # In case of an LLM error, be conservative: treat as not relevant,
        # and print the error for debugging. Up to you to change this behavior.
        print(f"LLM check failed: {e}")
        return False

def run_youtube_monitor():
    """Main watcher logic."""
    for name, channel_id in CHANNELS.items():
        videos = fetch_new_videos(channel_id)
        print(f"🔎 Checking channel: {name} ({len(videos)} new videos)")
        for video in videos:
            vid_id = video["id"]["videoId"]
            title = video["snippet"]["title"]
            description = video["snippet"].get("description", "")
            url = f"https://www.youtube.com/watch?v={vid_id}"

            # --- LLM semantic check ---
            if not llm_relevance_check(title, description):
                print(f"❌ Skipped (not list-type): {title}")
                time.sleep(3)  # polite pause between LLM calls
                continue

            # --- Avoid duplicates ---
            if videos_collection.find_one({"video_id": vid_id}):
                continue

            print(f"📹 New relevant video found: {title}")

            videos_collection.insert_one({
                "video_id": vid_id,
                "title": title,
                "url": url,
                "channel": name,
                "processed": False,
                "timestamp": datetime.utcnow()
            })

            # --- Extraction ---
            try:
                print(f"🔍 Extracting phone data from {title} ...")
                process_video(
                    video_id=vid_id,
                    title=title,
                    url=url
                )
                videos_collection.update_one({"video_id": vid_id}, {"$set": {"processed": True}})
                print("✅ Extraction complete!")
            except Exception as e:
                print(f"❌ Error extracting data from {title}: {e}")

            # Wait between video processing to avoid hitting API quota
            time.sleep(5)

if __name__ == "__main__":
    run_youtube_monitor()
