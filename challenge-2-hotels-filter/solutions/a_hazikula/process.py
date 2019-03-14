import pandas as pd
import numpy as np
import json


input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))

df0 = pd.read_pickle(data_args['staging_folder'] + '/filtered0.pkl')
df5 = pd.read_pickle(data_args['staging_folder'] + '/filtered5.pkl')
df10 = pd.read_pickle(data_args['staging_folder'] + '/filtered10.pkl')
df15 = pd.read_pickle(data_args['staging_folder'] + '/filtered15.pkl')
df20 = pd.read_pickle(data_args['staging_folder'] + '/filtered20.pkl')
df25 = pd.read_pickle(data_args['staging_folder'] + '/filtered25.pkl')
df30 = pd.read_pickle(data_args['staging_folder'] + '/filtered30.pkl')
df35 = pd.read_pickle(data_args['staging_folder'] + '/filtered35.pkl')
df40 = pd.read_pickle(data_args['staging_folder'] + '/filtered40.pkl')
df45 = pd.read_pickle(data_args['staging_folder'] + '/filtered45.pkl')
df50 = pd.read_pickle(data_args['staging_folder'] + '/filtered50.pkl')

out = []

for input_record in input_json:
  selected_df=eval('df%s' %str(int(input_record['stars']*10)))
  condition1 = (selected_df['price']>=input_record['min_price'])& (selected_df['price']<=input_record['max_price'])
  filtered_df =selected_df.loc[condition1]

  #filtered_df=selected_df.loc[(selected_df['price']>=input_record['min_price'])& (selected_df['price']<=input_record['max_price']),:]



  dist_arr = ((input_record['lon']-filtered_df['lon'].values) ** 2 + \
                      (input_record['lat']-filtered_df['lat'].values) ** 2)


  if not filtered_df.empty:
        closest_place = filtered_df.iloc[np.argmin(dist_arr),:].to_dict()
  else:
        closest_place = {'missing':True}

  out.append(closest_place.copy())

json.dump(out,open('output.json','w'))