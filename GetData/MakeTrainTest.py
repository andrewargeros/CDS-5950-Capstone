import pandas as pd
import os
import re
import shutil

from sklearn.model_selection import train_test_split

all_data = pd.read_csv('../all_data.csv')
classes = pd.read_csv('../beer_styles_classes.csv')

all_beers = os.listdir("../Beer Images")

all_data['num_usable_files'] = all_data['beer_id'].apply(
    lambda x: len([b for b in all_beers if b.startswith(str(x))]))

merged_styles = pd.merge(all_data, classes, on='style')
merged_styles.groupby('beer_style').sum().sort_values(
    'num_usable_files', ascending=False)[['num_files', 'num_usable_files']]

all_beers_df = pd.DataFrame(all_beers, columns=['file_name'])
all_beers_df['beer_id'] = all_beers_df['file_name'].apply(lambda x: re.sub(r"_.*$", "", x)).astype(int)
all_beers_df = pd.merge(all_beers_df, merged_styles, on = "beer_id", how='left')

X_train, X_test, y_train, y_test = train_test_split(all_beers_df.drop('beer_style', axis=1), all_beers_df['beer_style'], 
                                                    test_size=0.2, random_state=69)

X_train['beer_style'] = y_train
X_test['beer_style'] = y_test

styles = [re.sub(r"[^A-Za-z ].*$", "", x)
          for x in X_train.beer_style.unique().tolist()]

[os.mkdir(
    f"C:/PersonalScripts/CDS-5950-Capstone/Data/train/{x}") for x in styles]
[os.mkdir(
    f"C:/PersonalScripts/CDS-5950-Capstone/Data/test/{x}") for x in styles]

for row in X_train.itertuples():
  style = re.sub(r"[^A-Za-z ].*$", "", str(row.beer_style))
  folder = f"C:/PersonalScripts/CDS-5950-Capstone/Data/test/{style}"

  shutil.copy(
      f"C:/PersonalScripts/CDS-5950-Capstone/Beer Images/{row.file_name}", folder)

for row in X_test.itertuples():
  style = re.sub(r"[^A-Za-z ].*$", "", str(row.beer_style))
  folder = f"C:/PersonalScripts/CDS-5950-Capstone/Data/test/{style}"

  shutil.copy(
      f"C:/PersonalScripts/CDS-5950-Capstone/Beer Images/{row.file_name}", folder)
