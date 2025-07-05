import requests
import time
import io
from PIL import Image
import numpy as np
import torch
from transformers import AutoModel

# In the URL, the random letters and numbers is the UUID
MANGA_UUID = ""
# Add chapter number, later make it go until the last chapter
CHAPTER = ""

API_URL = "https://api.mangadex.org"

# Just to see that the model is being processed
print("Loading model")
# main model script 
model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()
print("Model loaded")

def get_chapter_id(manga_uuid, chapter_number):
    """Finds the chapter ID for a given manga and chapter number."""
    try:
        params = {
            "manga": manga_uuid,
            "chapters": [chapter_number],
            "translatedLanguage[]": ["en"]
        }
        response = requests.get(f"{API_URL}/chapter", params=params)
        response.raise_for_status()
        data = response.json()
        if data["data"]:
            return data["data"][0]["id"]
        else:
            print(f"Error: Chapter {chapter_number} not found.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching chapter list: {e}")
        return None

def get_page_urls(chapter_id):
    """Gets the full, downloadable image URLs for a chapter."""
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

# Execute program
print(f"Fetching Chapter {CHAPTER} for manga UUID: {MANGA_UUID}")
target_chapter_id = get_chapter_id(MANGA_UUID, CHAPTER)

if target_chapter_id:
    print(f"Found Chapter ID: {target_chapter_id}")
    page_urls = get_page_urls(target_chapter_id)

    if page_urls:
        print(f"Found {len(page_urls)} pages. Downloading images...")

        # create a list to hold all chapters images
        chapter_pages_for_model = []
        for url in page_urls:
            image_data = download_image_as_numpy(url)
            if image_data is not None:
                chapter_pages_for_model.append(image_data)
            # sleep to not overload api
            time.sleep(1)

        print("ALl images loaded. Feed to model...")

        # define character bank here
        character_bank = {"images": [], "names": []}

        # run model on all images
        with torch.no_grad():
            per_page_results = model.do_chapter_wide_prediction(
                    chapter_pages_for_model,
                    character_bank
            )

        # print the final transcript
        transcript = []
        for i, page_result in enumerate(per_page_results):
            transcript.append(f"\n--- Page {i + 1} ---")
            transcript.extend(page_result['transcript'])

        for line in transcript:
            print(line)
