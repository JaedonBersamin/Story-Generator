import requests
import io
import time
from PIL import Image
from transformers import pipeline

# --- 1. CONFIGURATION ---
MANGA_UUID = "a25e46ec-30f7-4db6-89df-cacbc1d9a900" # Example: "Solo Leveling"
CHAPTER_TO_FIND = "1"

# --- 2. SETUP MODELS AND API ---

API_URL = "https://api.mangadex.org"

# Load a dedicated pipeline for Image-to-Text OCR
# This model is simpler and better for direct transcription.
print("Loading OCR model...")
ocr_pipeline = pipeline("image-to-text", model="kha-white/manga-ocr-base")

# --- 3. API & HELPER FUNCTIONS ---

def find_chapter_id(manga_uuid, chapter_to_find):
    """Fetches all chapters for a manga and finds the ID for a specific one."""
    try:
        response = requests.get(
            f"{API_URL}/manga/{manga_uuid}/feed",
            params={"translatedLanguage[]": ["en"], "order[chapter]": "asc"}
        )
        response.raise_for_status()
        chapters = response.json().get("data", [])

        if not chapters:
            print("No chapters found for this manga.")
            return None

        for chapter in chapters:
            if chapter["attributes"]["chapter"] == chapter_to_find:
                return chapter["id"]

        print(f"Error: Could not find Chapter '{chapter_to_find}'.")
        print("Available chapters are:", [ch["attributes"]["chapter"] for ch in chapters])
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching chapter feed: {e}")
        return None

def get_page_urls(chapter_id):
    """Gets the full, downloadable image URLs for a given chapter ID."""
    try:
        response = requests.get(f"{API_URL}/at-home/server/{chapter_id}")
        response.raise_for_status()
        server_data = response.json()
        base_url = server_data["baseUrl"]
        chapter_hash = server_data["chapter"]["hash"]
        page_filenames = server_data["chapter"]["data"]
        image_urls = [
            f"{base_url}/data/{chapter_hash}/{filename}"
            for filename in page_filenames
        ]
        return image_urls
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page URLs: {e}")
        return []

# --- 4. MAIN EXECUTION ---

print(f"Searching for Chapter {CHAPTER_TO_FIND} in manga UUID: {MANGA_UUID}")
target_chapter_id = find_chapter_id(MANGA_UUID, CHAPTER_TO_FIND)

if target_chapter_id:
    print(f"Found Chapter ID: {target_chapter_id}")
    page_urls = get_page_urls(target_chapter_id)

    if page_urls:
        print(f"Found {len(page_urls)} pages. Transcribing now...")
        
        full_transcript = ""
        for i, url in enumerate(page_urls):
            print(f"\n--- Processing Page {i + 1} ---")
            try:
                # Download the image directly into memory
                response = requests.get(url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))

                # Feed the image to the OCR pipeline
                # The pipeline returns a list of dictionaries, we extract the 'generated_text'
                page_text_list = ocr_pipeline(image)
                
                # Consolidate all found text on the page
                page_text = "\n".join([item['generated_text'] for item in page_text_list])
                
                print(page_text)
                full_transcript += page_text + "\n"

            except Exception as e:
                print(f"Failed to process page: {e}")

            time.sleep(1) # Be respectful of the API

        print("\n\n--- COMPLETE TRANSCRIPT ---")
        print(full_transcript)
