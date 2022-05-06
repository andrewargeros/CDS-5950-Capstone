from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

options = ['a photo of a can or bottle', 'a photo of a glass of beer', 'a photo of a person or group of people', 'a photo of a company logo']

def classify_image(img, options, model = model, processor = processor):
  image = Image.open(img)
  inputs = processor(text=options, images=image, return_tensors="pt", padding=True)
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = logits_per_image.softmax(dim=1) # take softmax for probability

  return dict(zip(options, probs.tolist()[0]))

files = [file for file in os.listdir('/mnt/c/personalscripts/cds-5950-capstone/images') if file.endswith('.jpg')]

for img_path in tqdm(files):
  try:
    pred = classify_image(f"/mnt/c/personalscripts/cds-5950-capstone/images/{img_path}", options)
    print(img_path, pred)

    if pred['a photo of a glass of beer'] > 0.33:
      os.replace(f"/mnt/c/personalscripts/cds-5950-capstone/images/{img_path}", f"/mnt/c/personalscripts/cds-5950-capstone/Beer Images/{img_path}")
      print(f"{img_path} moved to Beer Images")
    else:
      print(f"{img_path} is not a beer - {max(pred, key=pred.get)}")
  except:
    print(f"Error with {img_path}")