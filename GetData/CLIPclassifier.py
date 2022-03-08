from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

options = ['a photo of a can or bottle', 'a photo of a glass of beer']

def classify_image(img, options, model = model, processor = processor):
  image = Image.open(img)
  inputs = processor(text=options, images=image, return_tensors="pt", padding=True)
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = logits_per_image.softmax(dim=1) # take softmax for probability

  return dict(zip(options, probs.tolist()[0]))

files = [file for file in os.listdir('/Images') if file.endswith('.jpg')]

for img_path in files:
  pred = classify_image(img_path, options)
  print(img_path, pred)

  if pred['a photo of a can or bottle'] > 0.5:
    os.replace(img_path, f"/content/cans/{img_path}")
  else:
    os.replace(img_path, f"/content/beer/{img_path}")