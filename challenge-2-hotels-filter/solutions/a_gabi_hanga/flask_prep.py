import json
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
from flask import current_app
import multiprocessing

app = Flask(__name__)

@app.route("/started")
def started():
    return 'DIC'
    
def smallproc(inp):
    
    global dic

    star = inp['stars']
    minp, maxp = inp['min_price'], inp['max_price']
    lon, lat = inp['lon'], inp['lat']

    try:
        lower = np.searchsorted(dic[star]['price'], minp, "left")
        upper = np.searchsorted(dic[star]['price'], maxp, "right")
    except:
         return {'missing': True}

    distances = np.array([(dic[star]['lon'][k] - lon) ** 2 + (dic[star]['lat'][k] - lat) ** 2 
                for k in range(lower, upper)])

    if len(distances) != 0:
        i = lower + np.argmin(distances)
        return {'lat': dic[star]['lat'][i],
                         'lon': dic[star]['lon'][i],
                         'name': dic[star]['name'][i],
                         'stars': star,
                         'price': dic[star]['price'][i]}
    else:
        return {'missing': True}
    
@app.route("/")
def process():
    
    input_json = json.load(open('input.json','r'))
    global dic
    dic = current_app.dic
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    out = list(pool.map(smallproc, input_json))
    json.dump(out, open('output.json', 'w'))
    return 'DIC'     
        
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

args = json.load(open('data.json','r'))
df = pd.read_csv(args['datafile']).drop_duplicates().set_index(["stars", "price"]).sort_index().reset_index("price")

df_dic = {}
for s in df.index.unique().tolist():
    df_dic[s] = {}
    for k in df.keys():
        df_dic[s][k] = np.array(df[k].loc[s].tolist())
        
app.dic = df_dic
dic = None

if __name__ == '__main__':
    app.run(debug = True, port = 5112)