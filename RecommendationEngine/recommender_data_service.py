# recommender_data_service.py

import re
import json
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from deep_translator import GoogleTranslator
from datetime import datetime
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))   
db = client["MobileDB"]                         
videos_collection = db["videos"]
phones_collection = db["phones"]

translator = GoogleTranslator()



def chunk_text(text, max_len=5000):
    """Split text into chunks of max_len chars."""
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

def fetch_transcript(video_id):
    """
    Fetch transcript for a YouTube video.
    First tries English, then any available language with translation.
    """
    try:
        ytt_api = YouTubeTranscriptApi()

        transcript = ytt_api.fetch(video_id, languages=['en'])

        transcript_data = transcript.to_raw_data()
        text = " ".join([entry['text'] for entry in transcript_data])
        return text, 'en'
    
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"No English transcript found for {video_id}")
        
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            
            for transcript in transcript_list:
                try:
                    fetched_transcript = transcript.fetch()
                    transcript_data = fetched_transcript.to_raw_data()
                    text = " ".join([entry['text'] for entry in transcript_data])
                    
                    # If it's not English, translate it
                    if transcript.language_code != 'en':
                        print(f"Translating from {transcript.language_code} to English")
                        # Translate in chunks to avoid API limits
                        chunks = chunk_text(text)
                        translated_chunks = []
                        
                        for chunk in chunks:
                            try:
                                translated = translator.translate(chunk, src=transcript.language_code, dest='en')
                                translated_chunks.append(translated)
                            except Exception as e:
                                print(f"Translation error for chunk: {e}")
                                translated_chunks.append(chunk)  
                        
                        text = " ".join(translated_chunks)
                    
                    return text, transcript.language_code
                    
                except Exception as e:
                    print(f"Error fetching transcript in {transcript.language_code}: {e}")
                    continue
            
            print(f"No transcripts available for video {video_id}")
            return None, None
            
        except Exception as e:
            print(f"Error listing transcripts for {video_id}: {e}")
            return None, None
    
    except Exception as e:
        print(f"Unexpected error fetching transcript for {video_id}: {e}")
        return None, None



def segment_transcript(transcript, video_title):
    """
    Use LLM to segment transcript and extract phone information.
    Processes the full transcript by chunking if necessary.
    """
    max_context_length = 12000 
    
    if len(transcript) <= max_context_length:
        return _process_single_transcript(transcript, video_title)
    else:
        return _process_chunked_transcript(transcript, video_title, max_context_length)



def _process_single_transcript(transcript, video_title):
    """Process a single transcript that fits within token limits."""
    prompt = f"""
    You are analyzing a YouTube transcript from a video titled "{video_title}".
    The transcript reviews multiple phones.

    Task:
    - Divide transcript into sections for each phone reviewed.
    - For each phone:
        - Extract the phone name (be specific, include brand and model)
        - Extract the price mentioned in the transcript (if any)
        - Calculate price_range: round UP the price to nearest 5000 increment (e.g., 21500 → 25000, 28000 → 30000)
        - Summarize its pros and cons into a single paragraph (3-5 sentences).
    - Return ONLY a valid JSON array with objects having keys: "phone_name", "description", "price_range".
    - price_range should be an integer (25000, 30000, etc.) or null if no price mentioned.
    - Do not include any text before or after the JSON.
    
    Examples:
    - If price is 21,500 or ₹21500 → price_range: 25000
    - If price is 28,000 or ₹28k → price_range: 30000  
    - If price is 15,999 → price_range: 20000
    - If no price mentioned → price_range: null
    
    Transcript:
    {transcript}
    """

    return _call_llm_and_parse(prompt)

def _process_chunked_transcript(transcript, video_title, max_length):
    """
    Process transcript in overlapping chunks to ensure no phones are missed.
    """
    chunk_size = max_length - 1000  
    overlap = 500 
    
    chunks = []
    start = 0
    
    while start < len(transcript):
        end = min(start + chunk_size, len(transcript))
        chunk = transcript[start:end]
        chunks.append(chunk)
        
        if end >= len(transcript):
            break
        start = end - overlap
    
    print(f"Processing transcript in {len(chunks)} chunks...")
    
    all_phones = []
    phone_names_seen = set() 
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        prompt = f"""
        You are analyzing part {i+1} of {len(chunks)} from a YouTube transcript titled "{video_title}".
        This transcript reviews multiple phones.

        Task:
        - Find all phones mentioned in this chunk.
        - For each phone:
            - Extract the phone name (be specific, include brand and model)
            - Extract the price mentioned in the transcript (if any)
            - Calculate price_range: round UP the price to nearest 5000 increment (e.g., 21500 → 25000, 28000 → 30000)
            - Summarize its pros and cons based on what's mentioned in this chunk (4-6 sentences).
        - Return ONLY a valid JSON array with objects having keys: "phone_name", "description", "price_range".
        - price_range should be an integer (25000, 30000, etc.) or null if no price mentioned.
        - Do not include any text before or after the JSON.
        - If no phones are clearly discussed in this chunk, return an empty array [].
        
        Examples:
        - If price is 21,500 or ₹21500 → price_range: 25000
        - If price is 28,000 or ₹28k → price_range: 30000  
        - If price is 15,999 → price_range: 20000
        - If no price mentioned → price_range: null
        
        Transcript chunk:
        {chunk}
        """
        
        chunk_phones = _call_llm_and_parse(prompt)
        
        # Add phones that haven't been seen before
        for phone in chunk_phones:
            phone_name = phone.get('phone_name', '').strip()
            if phone_name and phone_name not in phone_names_seen:
                phone_names_seen.add(phone_name)
                all_phones.append(phone)
    
    # Final consolidation pass - merge similar phone names and combine descriptions
    consolidated_phones = _consolidate_duplicate_phones(all_phones)
    
    print(f"Found {len(consolidated_phones)} unique phones after processing all chunks")
    return consolidated_phones


def infer_price_range_from_title(title: str):
    """Infer price range from video title (e.g. 20000_to_30000)."""
    title = title.lower()
    match = re.findall(r'(\d{2,3})k|\d{5}', title)

    if not match:
        under_match = re.search(r'under\s*(\d{2,3})k|\d{5}', title)
        if under_match:
            upper = int(re.sub(r'[^0-9]', '', under_match.group()))
            return f"0_to_{upper}"
        return None

    prices = []
    for p in match:
        if isinstance(p, tuple):
            p = next(filter(None, p), None)
        if not p:
            continue
        if len(p) <= 3:
            prices.append(int(p) * 1000)
        else:
            prices.append(int(p))

    if len(prices) == 1:
        return f"0_to_{prices[0]}"
    elif len(prices) >= 2:
        return f"{min(prices)}_to_{max(prices)}"
    return None



def _consolidate_duplicate_phones(phones):
    """
    Consolidate phones with similar names and merge their descriptions.
    Also handles price_range consolidation.
    """
    if not phones:
        return []
    
    consolidated = {}
    
    for phone in phones:
        phone_name = phone.get('phone_name', '').strip()
        description = phone.get('description', '').strip()
        price_range = phone.get('price_range')
        
        if not phone_name:
            continue
            
        # Simple similarity check - could be enhanced with fuzzy matching
        found_match = False
        for existing_name in consolidated.keys():
            # Check if phone names are similar (basic approach)
            if _are_phone_names_similar(phone_name, existing_name):
                # Merge descriptions
                existing_desc = consolidated[existing_name]['description']
                if description not in existing_desc:  # Avoid duplicate content
                    consolidated[existing_name]['description'] = f"{existing_desc} {description}".strip()
                
                # Update price_range if current one is null but new one has value
                if consolidated[existing_name]['price_range'] is None and price_range is not None:
                    consolidated[existing_name]['price_range'] = price_range
                
                found_match = True
                break
        
        if not found_match:
            consolidated[phone_name] = {
                'phone_name': phone_name,
                'description': description,
                'price_range': price_range
            }
    
    return list(consolidated.values())

def _are_phone_names_similar(name1, name2):
    """
    Simple similarity check for phone names.
    Could be enhanced with more sophisticated matching.
    """
    name1_clean = name1.lower().replace(' ', '')
    name2_clean = name2.lower().replace(' ', '')
    
    # Check if one name contains the other (for cases like "iPhone 15" vs "iPhone 15 Pro")
    return name1_clean in name2_clean or name2_clean in name1_clean

def _call_llm_and_parse(prompt):
    """
    Call the LLM with the given prompt and parse the JSON response.
    """
    try:
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        
        # Generate response
        response = llm.invoke(prompt)
        
        # Extract content from response
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to parse JSON
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        # Validate the structure
        if isinstance(data, list):
            valid_data = []
            for item in data:
                if isinstance(item, dict) and 'phone_name' in item and 'description' in item:
                    # Validate and clean price_range
                    price_range = item.get('price_range')
                    if price_range is not None:
                        try:
                            price_range = int(price_range)
                            # Ensure it's in 5000 increments and reasonable range
                            if price_range % 5000 != 0 or price_range < 5000 or price_range > 200000:
                                print(f"Invalid price_range {price_range} for {item.get('phone_name')}, setting to null")
                                price_range = None
                        except (ValueError, TypeError):
                            print(f"Invalid price_range format for {item.get('phone_name')}, setting to null")
                            price_range = None
                    
                    valid_item = {
                        'phone_name': item['phone_name'],
                        'description': item['description'],
                        'price_range': price_range
                    }
                    valid_data.append(valid_item)
            return valid_data
        else:
            print("LLM response is not a list")
            return []
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw content: {content[:200]}...")  # Show first 200 chars
        return []
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []

def process_video(video_id, title, url, price_range=None, uploaded_at=None):
    """
    Process a YouTube video: extract transcript, segment into phones, store in MongoDB.
    Automatically infers price range if not provided.
    """
    print(f"Processing video: {title}")

    # 🔹 Auto-infer price range if not given
    if not price_range:
        price_range = infer_price_range_from_title(title)
        print(f"💰 Inferred price range from title: {price_range}")

    # Get transcript
    transcript, lang = fetch_transcript(video_id)
    if not transcript:
        print(f"❌ Skipping video {video_id} - no transcript found")
        return None

    print(f"✅ Retrieved transcript in {lang}")

    # Segment transcript into phones
    phone_data = segment_transcript(transcript, title)

    # Store video metadata in MongoDB
    video_doc = {
        "youtube_id": video_id,
        "title": title,
        "url": url,
        "price_range": price_range,
        "uploaded_at": uploaded_at or datetime.utcnow(),
        "original_language": lang,
        "processed_at": datetime.utcnow()
    }

    try:
        videos_collection.update_one(
            {"youtube_id": video_id},
            {"$set": video_doc},
            upsert=True
        )
        print(f"✅ Stored video metadata for {title}")
    except Exception as e:
        print(f"Error storing video metadata: {e}")

    # Store phones
    if phone_data and isinstance(phone_data, list):
        for entry in phone_data:
            phone_doc = {
                "video_id": video_id,
                "phone_name": entry.get("phone_name", "Unknown"),
                "description": entry.get("description", ""),
                "price_range": entry.get("price_range"),
                "video_price_range": price_range,  # 🔹 consistent
                "created_at": datetime.utcnow()
            }

            try:
                phones_collection.update_one(
                    {"video_id": video_id, "phone_name": phone_doc["phone_name"]},
                    {"$set": phone_doc},
                    upsert=True
                )
            except Exception as e:
                print(f"Error storing phone data: {e}")

        print(f"✅ Stored {len(phone_data)} phones from video {title}")
    else:
        print("⚠️ No valid phone data extracted from transcript")


# Example usage
if __name__ == "__main__":
    process_video(
        video_id="clc5WoFWNyU",
        title="Best Paisa Wasool Phones for You 20k to 30k 🔥 In Box Packed Category - My Top Picks",
        url="https://youtu.be/clc5WoFWNyU?si=B-sly8l7XLdooc2I"
    )