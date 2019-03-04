import pandas as pd
import json

input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))

df = pd.read_pickle(data_args['staging_folder'] + '/filtered.pkl')
df2 = pd.DataFrame.from_dict(input_json, orient = "columns")

out = []
    
for i in df2.index:
    df['distance'] = ((df2.get_value(i, 'lon')-df['lon']) **2 + (df2.get_value(i, 'lat')-df['lat'] **2))
    out.append(df.ix[df['distance'].idxmin(),["lon","lat","name"]].to_dict().copy())

json.dump(out,open('output.json','w'))
