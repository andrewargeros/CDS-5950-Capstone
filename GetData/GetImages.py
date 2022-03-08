import requests
import re
import shutil
import pandas as pd

from tqdm import tqdm
from uuid import uuid4
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def make_headers():
  headers = {
    'User-Agent': uuid4().hex,
    'From': f"{uuid4().hex}@{uuid4().hex}.com"
  }
  return headers

def get_images(link, headers, outpath):
  try:
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    response = session.get(link, headers=headers)
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
        img_file = session.get(img_link).content
        with open(f"{outpath}/{beer_id}_{i}.jpg", "wb+") as f:
          f.write(img_file)
        i += 1

    info['num_files'] = i
    return info
  except:
    print("Error")


def get_beer_urls(base_link, headers):
  session = requests.Session()
  retry = Retry(connect=3, backoff_factor=0.5)
  adapter = HTTPAdapter(max_retries=retry)
  session.mount('http://', adapter)
  session.mount('https://', adapter)

  lsoup = BeautifulSoup(session.get(base_link, headers=headers).content)

  beers = lsoup.find_all('div', {'class': 'beer-details'})

  beer_urls = []
  for beer in beers:
    links = beer.find_all('a', href=True)
    style_link = [link.get('href')
                  for link in links if link.get('href').startswith('/b/')]

    beer_urls.append(style_link[0])
  
  return beer_urls

res2 = requests.get('https://untappd.com/beer/top_rated', headers=make_headers())
soup2 = BeautifulSoup(res2.content)
opts = soup2.find_all('select', {'aria-label':'Style picker'})[0].find_all('option')

beer_styles = [opt.get('data-value-slug')
               for opt in opts if opt.get('data-value-slug') is not None]

base_link = "https://untappd.com/beer/top_rated?type="

all_beer_urls = []

for slug in tqdm(beer_styles):
  base_link = f"https://untappd.com/beer/top_rated?type={slug}"
  all_beer_urls.append(get_beer_urls(base_link, make_headers()))

all_beer_urls = [item for sublist in all_beer_urls for item in sublist]

print(f"Found {len(all_beer_urls)} beers")

all_data = []
for url in tqdm(all_beer_urls):
  page_link = 'https://untappd.com' + url + "/photos"
  data = get_images(page_link, make_headers(), './Images')
  all_data.append(data)
  print(data)

shutil.make_archive('Image_archive', 'zip', './Images')
pd.DataFrame(all_data).to_csv('./all_data.csv', index=False)
