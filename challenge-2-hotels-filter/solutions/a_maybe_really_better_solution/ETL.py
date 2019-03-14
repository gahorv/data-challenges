import pandas as pd
import json

args = json.load(open('data.json','r'))

df = pd.read_csv(args['datafile'])

df.drop_duplicates().to_pickle(args['staging_folder'] + '/filtered.pkl')

