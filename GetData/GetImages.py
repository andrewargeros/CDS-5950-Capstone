import requests
import re
import shutil
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'My User Agent 1.69454',
    'From': 'youremail@domai55n.com'
}

def get_images(link, headers, outpath):
  try:
    response = requests.get(link, headers=headers)
    beer_id = re.findall('/\d+/', link)[0].replace('/', '')

    soup = BeautifulSoup(response.content)
    name = soup.find_all('h1')[0].text.strip()
    style = soup.find_all('p', {"class": "style"})[0].text.strip().upper()

    info = {'name': name,
            'style': style,
            'beer_id': beer_id}

    i = 1
    for img in soup.find_all('img'):
      img_link = img.get('data-original')
      if img_link is not None:
        print(img_link)
        img_file = requests.get(img_link).content
        with open(f"{outpath}/{beer_id}_{i}.jpg", "wb+") as f:
          f.write(img_file)
        i += 1

    info['num_files'] = i
    return info
  except:
    print("Error")


res2 = requests.get('https://untappd.com/beer/top_rated', headers=headers)
soup2 = BeautifulSoup(res2.content)
opts = soup2.find_all('select', {'aria-label':'Style picker'})[0].find_all('option')

beer_styles = [opt.get('data-value-slug')
               for opt in opts if opt.get('data-value-slug') is not None]

base_link = "https://untappd.com/beer/top_rated?type="

beer_urls = []

for slug in beer_styles:
  base_link = f"https://untappd.com/beer/top_rated?type={slug}"
  lsoup = BeautifulSoup(requests.get(base_link, headers=headers).content)

  beers = lsoup.find_all('div', {'class': 'beer-details'})

  for beer in beers:
    links = beer.find_all('a', href=True)
    style_link = [link.get('href')
                  for link in links if link.get('href').startswith('/b/')]

    beer_urls.append(style_link[0])

all_data = []
for url in tqdm(beer_urls):
  page_link = 'https://untappd.com' + url + "/photos"
  data = get_images(page_link, headers, '/content/Images')
  all_data.append(data)
  print(data)

shutil.make_archive('Image_archive', 'zip', './Images')
pd.DataFrame(all_data).to_csv('./all_data.csv', index=False)
