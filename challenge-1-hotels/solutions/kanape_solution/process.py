import pandas as pd
import numpy as np
import json


input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))

df = pd.read_csv(data_args['staging_folder'] + '/filtered.csv')

out = []

for place in input_json:
    
    df['distance'] = ((place['lon']-df['lon']) ** 2 + (place['lat']-df['lat']) ** 2) ** 0.5
    
 #   closest_place = df.ix[df['distance'].idxmin(),["lon","lat","name"]].to_dict()
    
 #   out.append(closest_place.copy())
    out.append(df.ix[df['distance'].idxmin(),["lon","lat","name"]].to_dict().copy())

json.dump(out,open('output.json','w'))
