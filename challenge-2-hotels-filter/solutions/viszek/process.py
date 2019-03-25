import pandas as pd
import numpy as np
import json
import multiprocessing

input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))

df = pd.read_pickle(data_args['staging_folder'] + '/filtered.pkl')

#out = []

def distcalc(input_record):

    filtered_df = df.loc[(df['stars'] == input_record['stars']) &
           (df['price'] >= input_record['min_price']) &
           (df['price'] <= input_record['max_price']),:]
   
    dist_arr = ((input_record['lon']-filtered_df['lon'].values) ** 2 + \
                      (input_record['lat']-filtered_df['lat'].values) ** 2)

    if not filtered_df.empty:
        closest_place = filtered_df.iloc[np.argmin(dist_arr),:].to_dict()
    else:
        closest_place = {'missing':True}

    #out.append(closest_place.copy())
    return closest_place

p = multiprocessing.Pool(processes = multiprocessing.cpu_count())
out = p.map(distcalc,input_json)

json.dump(out,open('output.json','w'))

