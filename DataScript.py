import requests
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


