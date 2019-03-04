import pandas as pd
import json

args = json.load(open('data.json','r'))

df = pd.read_csv(args['datafile'])

df = df.sort_values(by=['lon']).loc[:,["lon","lat","name"]]
df = df.drop_duplicates(subset=['name'], keep='first')
df.reset_index(inplace = True, drop=True)

df[['lon','lat','name']].to_pickle(args['staging_folder'] + '/filtered.pickle')
