import json
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
from flask import current_app
app = Flask(__name__)

@app.route("/started")
def started():
    return 'FING'


@app.route("/")
def process():
    input_json = json.load(open('input.json','r'))
    input_df = pd.read_pickle(input_json['data_pickle'])
    
    #input_df
    
    #result = current_app.
    
    out = [{'most-used-formation':input_df[['home_formation','away_formation']].unstack().value_counts().index[0]}]
    
    
    json.dump(out,open('output.json','w'))
    return 'FING'


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

app.p_i_df = pd.read_csv(args['p_info'])
app.t_i_df = pd.read_csv(args['t_info'])
app.p_v_df = pd.read_csv(args['p_values'])
app.m_o_df = pd.read_csv(args['m_odds'])



if __name__ == '__main__':
    app.run(debug=True,port=5112)
