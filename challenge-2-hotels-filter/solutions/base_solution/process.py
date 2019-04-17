import pandas as pd
import numpy as np
import json


input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))

df = pd.read_csv(data_args['staging_folder'] + '/filtered.csv')

out = []

for input_record in input_json:
    
    df['distance'] = ((input_record['lon']-df['lon']) ** 2 + \
                      (input_record['lat']-df['lat']) ** 2) ** 0.5
    
    filtered_distances = df.loc[(df['stars'] == input_record['stars']) &
           (df['price'] >= input_record['min_price']) &
           (df['price'] <= input_record['max_price']),
           'distance']

    if not filtered_distances.empty:
        closest_place = df.loc[filtered_distances.idxmin(),
                            ['lat','lon','name','stars','price']].to_dict()
    else:
        closest_place = {'missing':True}

    out.append(closest_place.copy())

json.dump(out,open('output.json','w'))
