import pandas as pd
import numpy as np
import json


input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))

df = pd.read_pickle(data_args['staging_folder'] + '/filtered.pkl')

out = []

for input_record in input_json:
    
    filtered_df = df.loc[(df['stars'] == input_record['stars']) &
           (df['price'] >= input_record['min_price']) &
           (df['price'] <= input_record['max_price']),:]
   
    filtered_df['distance'] = ((input_record['lon']-filtered_df['lon']) ** 2 + \
                      (input_record['lat']-filtered_df['lat']) ** 2)

    if not filtered_df.empty:
        closest_place = df.loc[filtered_df['distance'].idxmin(),
                            ['lat','lon','name','stars','price']].to_dict()
    else:
        closest_place = {'missing':True}

    out.append(closest_place.copy())

json.dump(out,open('output.json','w'))
