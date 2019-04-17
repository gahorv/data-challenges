import pandas as pd
import json

args = json.load(open('data.json','r'))
data = pd.read_csv(args['datafile'])

data.drop_duplicates().reset_index(drop = True)[['lon','lat','name']].to_csv(args['staging_folder'] + '/filtered.csv', index = None)