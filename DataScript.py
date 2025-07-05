import requests
import io
import time
from PIL import Image
import numpy as np
import torch
from transformers import AutoModel

MANGA_UUID = "a25e46ec-30f7-4db6-89df-cacbc1d9a900"
CHAPTER_TO_FIND = "1"  # The chapter number you want to get.

# --- 2. SETUP MODELS AND API ---

API_URL = "https://api.mangadex.org"

# Load the pretrained Magiv2 model.
# Use .cuda() if you have a GPU for faster processing, otherwise remove it.
print("Loading transcription model...")
model = AutoModel.from_pretrained(
    "ragavsachdeva/magiv2",
    trust_remote_code=True
).cuda().eval()

# --- 3. API & HELPER FUNCTIONS ---

def find_chapter_id(manga_uuid, chapter_to_find):
    """
    Fetches all chapters for a manga and finds the ID for a specific one.
    This is more reliable than asking for a chapter number directly.
    """
    try:
        # The /feed endpoint gets the full chapter list for a manga.
        response = requests.get(
            f"{API_URL}/manga/{manga_uuid}/feed",
            params={"translatedLanguage[]": ["en"], "order[chapter]": "asc"}
        )
        response.raise_for_status()
        chapters = response.json().get("data", [])

        if not chapters:
            print("No chapters found for this manga.")
            return None

        # Loop through the list to find the chapter you want.
        for chapter in chapters:
            if chapter["attributes"]["chapter"] == chapter_to_find:
                print(f"Successfully found Chapter {chapter_to_find}.")
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

        # debugging
        #print("--- DEBUG: API Server Response ---")
        #print(server_data)

        base_url = server_data["baseUrl"]
        chapter_hash = server_data["chapter"]["hash"]
        page_filenames = server_data["chapter"]["data"]  # For high-quality images.

        image_urls = [
            f"{base_url}/data/{chapter_hash}/{filename}"
            for filename in page_filenames
        ]
        return image_urls
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page URLs: {e}")
        return []

def download_image_as_numpy(url):
    """Downloads an image from a URL and returns it as a NumPy array."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("L").convert("RGB")
        return np.array(image)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image {url}: {e}")
        return None

# --- 4. MAIN EXECUTION ---

print(f"Searching for Chapter {CHAPTER_TO_FIND} in manga UUID: {MANGA_UUID}")
target_chapter_id = find_chapter_id(MANGA_UUID, CHAPTER_TO_FIND)

if target_chapter_id:
    print(f"Found Chapter ID: {target_chapter_id}")
    page_urls = get_page_urls(target_chapter_id)

    if page_urls:
        print(f"Found {len(page_urls)} pages. Downloading images into memory...")

        # Create a list to hold all the chapter's images.
        chapter_pages_for_model = []
        for url in page_urls:
            image_data = download_image_as_numpy(url)
            if image_data is not None:
                chapter_pages_for_model.append(image_data)
            # Be respectful of the API and don't send requests too quickly.
            time.sleep(1)

        print("All images loaded into memory. Feeding them to the model...")

        # NOTE: You can define your character bank here if needed for the model.
        # For a generic first pass, an empty bank is fine.
        character_bank = {"images": [], "names": []}

        # Run the model on the entire list of images.
        per_page_results = model.do_chapter_wide_prediction(
            chapter_pages_for_model,
            character_bank
        )

       # --- Print the final transcript from the model's output ---
        print("\n--- GENERATED TRANSCRIPT ---")
        full_transcript = []
        for i, page_result in enumerate(per_page_results):
            # --- DEBUGGING: Uncomment the line below to see what the model returns for each page ---
            # print(f"Raw output for Page {i + 1}: {page_result}")

            full_transcript.append(f"\n--- Page {i + 1} ---")

            # --- THE FIX ---
            # Use .get('transcript', []) to safely get the transcript.
            # If the 'transcript' key doesn't exist, it returns an empty list []
            # instead of crashing.
            page_transcript = page_result.get('transcript', [])
            full_transcript.extend(page_transcript)

        for line in full_transcript:
            print(line)
