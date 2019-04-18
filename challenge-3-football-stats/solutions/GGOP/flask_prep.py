#coding: utf-8
import json
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
from flask import current_app
from datetime import datetime
import time
from multiprocessing import Pool, cpu_count
import Levenshtein



app = Flask(__name__)

@app.route("/started")
def started():
    return 'FING'


@app.route("/")
def process():
    input_json = json.load(open('input.json','r'))
    m_i_df = pd.read_pickle(input_json['data_pickle'])
    
    #input_df
    
    #result = current_app.

    print('proc till tolist')
    m_i_df["away_players"] = m_i_df[['away1_id','away2_id','away3_id','away4_id',"away5_id","away6_id","away7_id",'away8_id','away9_id','away10_id','away11_id']].values.tolist()
    m_i_df["home_players"] = m_i_df.loc[:,['home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id']].values.tolist()
    m_i_df["score_home"], m_i_df["score_away"] = m_i_df["score"].str.split('-', 1).str
    m_i_df["score_home"] = m_i_df["score_home"].apply(int)
    m_i_df["score_away"] = m_i_df["score_away"].apply(int)
    m_i_df["away_players"] = m_i_df[['away1_id','away2_id','away3_id','away4_id',"away5_id", "away6_id","away7_id",'away8_id','away9_id','away10_id','away11_id']].values.tolist()
    m_i_df["home_players"] = m_i_df.loc[:,['home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id']].values.tolist()
    m_i_df["away_players_bd"] = m_i_df["away_players"].apply(eletkor_to_list)
    m_i_df["home_players_bd"] = m_i_df["home_players"].apply(eletkor_to_list)
    m_i_df["away_players_bd"] = m_i_df[['date','away_players_bd',]].values.tolist()
    m_i_df["home_players_bd"] = m_i_df[['date','home_players_bd',]].values.tolist()
    m_i_df["avg_home_age"] = m_i_df["home_players_bd"].apply(calculate_age)
    m_i_df["avg_away_age"] = m_i_df["away_players_bd"].apply(calculate_age)
    m_i_df["max_age"] = m_i_df["home_players_bd"].apply(calculate_max_age)
    m_i_df["max_age_place"] = m_i_df["home_players_bd"].apply(calculate_max_age_place)
    m_i_df["AVG_HEIGHT_H"] = m_i_df["home_players"].apply(avg_height)
    m_i_df["AVG_HEIGHT_A"] = m_i_df["away_players"].apply(avg_height)
    m_i_df["stad_cap_a"] = m_i_df.away_team.apply(stad_cap)
    m_i_df["stad_cap_h"] = m_i_df.home_team.apply(stad_cap)
    m_i_df["all_goals"] = m_i_df["score_away"]+m_i_df["score_home"]
    m_i_df["goal_difference"] =   m_i_df["score_away"]-m_i_df["score_home"]
    m_i_df_proba = pd.concat([m_i_df[:], m_i_df["away_players"].apply(height_to_list).apply(pd.Series).rename(columns = lambda x : 'heights_a_' + str(x+1))[:]], axis =1)
    m_i_df_proba = pd.concat([m_i_df_proba[:], m_i_df["home_players"].apply(height_to_list).apply(pd.Series).rename(columns = lambda x : 'heights_h_' + str(x+1))[:]], axis =1)
    pos_height_df = pd.lreshape(m_i_df_proba, {'position':['away1_pos','away2_pos','away3_pos','away4_pos',"away5_pos",
        "away6_pos","away7_pos",'away8_pos','away9_pos','away10_pos','away11_pos',
        'home1_pos','home2_pos','home3_pos','home4_pos','home5_pos','home6_pos','home7_pos',
         'home8_pos','home9_pos','home10_pos','home11_pos'],
                       'heights':['heights_a_1','heights_a_2','heights_a_3','heights_a_4','heights_a_5','heights_a_6',"heights_a_7",
        "heights_a_8","heights_a_9",'heights_a_10','heights_a_11',
        'heights_h_1','heights_h_2','heights_h_3','heights_h_4','heights_h_5','heights_h_6',"heights_h_7",
        "heights_h_8","heights_h_9",'heights_h_10','heights_h_11']}).pipe(lambda x: x[["position","heights"]])
    pos_height_df["heights"] = pos_height_df["heights"].apply(try_height_2)
    m_i_df = pd.concat([m_i_df[:],  m_i_df["away_players_bd"].apply(calculate_valamelyik_age).apply(pd.Series).rename(columns = lambda x : 'ages_a_' + str(x+1))[:]], axis =1)
    m_i_df = pd.concat([m_i_df[:], m_i_df["home_players_bd"].apply(calculate_valamelyik_age).apply(pd.Series).rename(columns = lambda x : 'ages_h_' + str(x+1))[:]], axis =1)
    pos_age_df = pd.lreshape(m_i_df, {'position':['away1_pos','away2_pos','away3_pos','away4_pos',"away5_pos",
        "away6_pos","away7_pos",'away8_pos','away9_pos','away10_pos','away11_pos',
        'home1_pos','home2_pos','home3_pos','home4_pos','home5_pos','home6_pos','home7_pos',
         'home8_pos','home9_pos','home10_pos','home11_pos'],
                       'ages':['ages_a_1','ages_a_2','ages_a_3','ages_a_4','ages_a_5','ages_a_6',"ages_a_7",
        "ages_a_8","ages_a_9",'ages_a_10','ages_a_11',
        'ages_h_1','ages_h_2','ages_h_3','ages_h_4','ages_h_5','ages_h_6',"ages_h_7",
        "ages_h_8","ages_h_9",'ages_h_10','ages_h_11']}).pipe(lambda x: x)[["position","ages"]]

    print("EDDIG ELJUT - PROCESS")


    new_df = pd.merge(m_i_df[["away_team","home_team","date","avg_home_age","avg_away_age","result","score"]]\
    ,current_app.m_o_df[["odds","odds1" ,"odds2" ,"oddsx","team_id_at","team_id_ht","date","score"]]\
    ,how='left', left_on=["away_team","home_team","date","score"], right_on = ["team_id_at","team_id_ht","date","score"])


    new_df[["odds","odds1" ,"odds2" ,"oddsx"]] = new_df[["odds","odds1" ,"odds2" ,"oddsx"]].applymap(oddsx_to_float)
    new_df = new_df.dropna(subset=['odds'])

    new_df["result"] = new_df["result"].apply(lambda x: float(x))

    new_df["win_as_udog_h"] = new_df.loc[:,["odds1" ,"odds2","result"]].apply(win_udog_h,axis=1)
    new_df["win_as_udog_a"] = new_df.loc[:,["odds1" ,"odds2","result"]].apply(win_udog_a,axis=1)

    new_df["lose_fav_h"] = new_df.loc[:,["odds1" ,"odds2","result"]].apply(lose_fav_h,axis=1)
    new_df["lose_fav_a"] = new_df.loc[:,["odds1" ,"odds2","result"]].apply(lose_fav_a,axis=1)

    new_df["looser"] = new_df.loc[:,["team_id_ht" ,"team_id_at","result"]].apply(looser,axis=1)
    new_df["looser_odds"] = new_df.loc[:,["odds1" ,"odds2","result"]].apply(looser_odds,axis=1)

    m_i_df["away_values"] = m_i_df.loc[:,["date" ,"away_players"]].apply(player_value,axis=1)

    m_i_df["home_values"] = m_i_df.loc[:,["date" ,"home_players"]].apply(player_value,axis=1)


    new_df = pd.merge(new_df, m_i_df[["away_values" ,"home_values","away_team","home_team","date","score"]],
                  how='left', 
                  left_on=["away_team","home_team","date","score"],
                  right_on = ["away_team","home_team","date","score"])
    

    value_a = m_i_df['away_values'].apply(pd.Series)
    value_h = m_i_df['home_values'].apply(pd.Series)

    value_a = value_a.rename(columns = lambda x : 'value_a_' + str(x+1))
    value_h = value_h.rename(columns = lambda x : 'value_h_' + str(x+1))






















    out = [
         {'most-used-formation':m_i_df[['home_formation','away_formation']].unstack().value_counts().index[0]},
          {'number-of-players-with-no-games':str(len(current_app.p_i_df)-len(current_app.p_i_df.loc[current_app.p_i_df.playerid.isin(pd.DataFrame(m_i_df[['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id","away6_id","away7_id",'away8_id','away9_id','home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id']].unstack().unique())[0])]))},

          {'player-with-highest-number-of-games':str(m_i_df[['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id","away6_id","away7_id",'away8_id','away9_id','home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id']].unstack().value_counts().index[0])},
          {'player-with-highest-number-of-games-where-his-team-didnt-concede': int(pd.concat([pd.DataFrame(m_i_df.loc[(m_i_df['score_home'] == 0)][['home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id']]),pd.DataFrame(m_i_df.loc[(m_i_df['score_away'] == 0)][['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id", "away6_id","away7_id",'away8_id','away9_id']])]).unstack().value_counts().index[0])},
          {'most-games-played-in-same-position-by-player':str(
            pd.lreshape(m_i_df, {'player_id':['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id","away6_id","away7_id",'away8_id','away9_id','home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id'],
                     'position':['away10_pos','away11_pos','away1_pos','away2_pos','away3_pos','away4_pos',"away5_pos","away6_pos","away7_pos",'away8_pos','away9_pos','home1_pos','home2_pos','home3_pos','home4_pos','home5_pos','home6_pos','home7_pos','home8_pos','home9_pos','home10_pos','home11_pos']})\
.pipe(lambda x:x)[["player_id","position"]]\
.groupby(pd.lreshape(m_i_df, {'player_id':['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id", "away6_id","away7_id",'away8_id','away9_id','home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id'],
                              'position':['away10_pos','away11_pos','away1_pos','away2_pos','away3_pos','away4_pos',"away5_pos","away6_pos","away7_pos",'away8_pos','away9_pos','home1_pos','home2_pos','home3_pos','home4_pos','home5_pos','home6_pos','home7_pos','home8_pos','home9_pos','home10_pos','home11_pos']})\
.pipe(lambda x:x)[["player_id","position"]].columns.tolist(),as_index=False).size().max())},

          {'most-different-positions-by-player':str(pd.lreshape(m_i_df, {'player_id':['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id",
       "away6_id","away7_id",'away8_id','away9_id','home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id',
        'home8_id','home9_id','home10_id','home11_id'],
                      'position':['away10_pos','away11_pos','away1_pos','away2_pos','away3_pos','away4_pos',"away5_pos",
       "away6_pos","away7_pos",'away8_pos','away9_pos',
       'home1_pos','home2_pos','home3_pos','home4_pos','home5_pos','home6_pos','home7_pos',
        'home8_pos','home9_pos','home10_pos','home11_pos']}).pipe(lambda x:x)[["player_id","position"]][["player_id","position"]].groupby('player_id')["position"].nunique().max())},     
          {'most-different-formations-by-player':str(pd.lreshape(m_i_df, {'player_id':['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id","away6_id","away7_id",'away8_id','away9_id','home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id','home8_id','home9_id','home10_id','home11_id'],'formation':['away_formation','away_formation','away_formation','away_formation','away_formation','away_formation','away_formation','away_formation','away_formation','away_formation','away_formation',"home_formation","home_formation","home_formation","home_formation","home_formation","home_formation","home_formation","home_formation","home_formation","home_formation","home_formation"]}).pipe(lambda x: x)[["player_id","formation"]].groupby('player_id')["formation"].nunique().max())},
          
          {'largest-odds-overcome-in-game':new_df[new_df["result"]!=0.0]["odds"].max()},
          {'largest-height-difference-overcome-in-game':m_i_df.loc[:,["result","AVG_HEIGHT_H","AVG_HEIGHT_A"]].apply(height_diff_OC,axis=1).max()},

          {'longest-time-in-days-between-two-games-for-player':None},
          
          {'biggest-value-difference':str(m_i_df.loc[:,["away_values" ,"home_values"]].apply(values_diff,axis=1).max())},
          
          {'biggest-value-difference-upset':int(abs(m_i_df.loc[:,["away_values" ,"home_values","result"]].apply(values_diff_ups,axis=1).min()))}, # an upset means the unexpected team won
          
          {'biggest-value-difference-with-higher-odds':int(new_df.loc[:,["away_values" ,"home_values","odds1","odds2"]].apply(values_diff_ups_odds,axis=1).max())},
          
          {'biggest-stadium-capacity-difference-upset':None},
          
          {'capacity-of-stadium-of-team-with-most-games':pd.DataFrame(pd.lreshape(m_i_df, {'team_id':['away_team','home_team'],'seats':["stad_cap_a",'stad_cap_h']}).groupby(pd.lreshape(m_i_df, {'team_id':['away_team','home_team'],'seats':["stad_cap_a",'stad_cap_h']})[["team_id","seats"]].columns.tolist(),as_index=False).size()).idxmax()[0][1]},
          
          {'id-of-oldest-team-to-win-a-game':id_of_oldest_team_to_win_a_game(m_i_df)},
          
          {'biggest-age-difference-between-teams-match-id':int(m_i_df.iloc[abs(m_i_df["avg_away_age"]-m_i_df["avg_home_age"]).idxmax(),:]["mkey"])},
          
          {'median-of-winning-team-average-age':(m_i_df.loc[:,["result","avg_away_age","avg_home_age"]].apply(gyoztes_kor,axis=1)).median()},
          
          {'median-of-favorite-team-average-age':int(new_df.loc[:,["odds1" ,"odds2","avg_home_age","avg_away_age"]].apply(fav_age,axis=1).median())}, # favorite means has lower odds of winning
          
          {'median-of-underdog-team-average-age':int(new_df.loc[:,["odds1" ,"odds2","avg_home_age","avg_away_age"]].apply(udog_age,axis=1).median())}, # underdog means has higher odds of winning
          
          {'team-with-most-wins-as-underdog':pd.lreshape(new_df, {'team_id':['team_id_at','team_id_ht'],'wins_as_udog':["win_as_udog_a",'win_as_udog_h']}).pipe(lambda x: x[["team_id","wins_as_udog"]])\
           .groupby("team_id").agg({"wins_as_udog":"sum"})["wins_as_udog"].idxmax()},
          
          {'team-with-most-losses-as-favorite':pd.lreshape(new_df, {'team_id':['team_id_at','team_id_ht'],'lose_as_fav':["lose_fav_a",'lose_fav_h']}).pipe(lambda x: x[["team_id","lose_as_fav"]])\
           .groupby("team_id").agg({"lose_as_fav":"sum"})["lose_as_fav"].idxmax()},
          
          {'team-with-lowest-average-odds-of-draw':pd.lreshape(new_df, {'team_id':['team_id_at','team_id_ht'],'oddsx':["oddsx",'oddsx']}).pipe(lambda x: x[["team_id","oddsx"]])\
           .groupby("team_id").agg({"oddsx":"mean"})["oddsx"].idxmin()},
          
          {'position-with-highest-average-value':None},
          
          {'position-with-largest-average-height':pos_height_df.groupby("position").agg({"heights":"mean"})["heights"].idxmax()},
          
          {'position-with-youngest-average-age':pos_age_df.groupby("position").agg({"ages":"mean"})["ages"].idxmin()},
          
          {'goalkeeper-with-most-clean-sheets':None},#az átlagosan legtöbb gólt kapó kapus születési dátuma
          
          {'stadium-capactiy-of-team-with-most-avg-goals-in-a-game':None},#átlagosan leggólgazdagabb meccseket játszó csapat stadionjának befogadóképessége
          
          {'team-with-highest-profit-for-losing':int(new_df[["looser_odds","looser"]].groupby("looser")["looser_odds"].sum().idxmax())},#a csapat, akinek, ha minden meccsén ellenük fogadsz, összesítve a legnagyobb profitot termeli (mindig ugyanakkora összeggel fogadsz rá)
          
          {'largest-std-in-goal-difference-team':int(pd.lreshape(m_i_df, {'team_id':['away_team','home_team'],
                                'goal_difference':["goal_difference",'goal_difference']}).pipe(lambda x: x[["team_id","goal_difference"]])\
          .groupby("team_id").agg({"goal_difference":"std"})["goal_difference"].idxmax())},#a legnagyobb gólkülönbség szórással rendelkező csapat
          
          {'player-with-most-different-teams': int(pd.lreshape(m_i_df, {'player_id':['away10_id','away11_id','away1_id','away2_id','away3_id','away4_id',"away5_id",
                 "away6_id","away7_id",'away8_id','away9_id',
                 'home1_id','home2_id','home3_id','home4_id','home5_id','home6_id','home7_id',
                  'home8_id','home9_id','home10_id','home11_id'],
                                'team_id':['away_team','away_team','away_team','away_team','away_team','away_team','away_team','away_team','away_team','away_team','away_team',
                                           'home_team','home_team','home_team','home_team','home_team','home_team','home_team','home_team','home_team','home_team','home_team']})\
              .groupby("player_id")["team_id"].nunique().idxmax())},#a legtöbb csapatban pályára lépő játékos
          
          {'longest-losing-streak-team':None},#a leghosszabb vesztes széria by team
          
          {'longest-home-winning-streak-stadium-capacity':None },#leghosszabb hazai pályás győzelmi széria helyszínének befogadóképessége
          
          {'win-ratio-of-actual-highest-rated-player':None},#az adott időpillanatban legértékesebb játékos átlagos win ratioja
          
          {'oldest-player-to-win-a-home-game':m_i_df.iloc[m_i_df[m_i_df["result"]==1]["max_age"].idxmax(),:]["home_players"][m_i_df.iloc[m_i_df[m_i_df["result"]==1]["max_age"].idxmax(),:]["max_age_place"][0]]}#a legidősebb hazai pályán győztes meccset játszó játékos
          ]
    

    
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
app.p_i_dict = app.p_i_df.set_index('playerid').T.to_dict('dict')
app.t_i_dict = app.t_i_df.set_index('team_id').T.to_dict('dict')




def eletkor_to_list(x):
    
    lista = []
    for i in x:
        
        try: 
        
            asd = current_app.p_i_dict[i]["dob"]
            lista.append(asd)
            
        except:
            continue 
        
    return lista

def calculate_age(double_list):
    
    match = double_list[0]
    today = datetime.strptime(match, '%Y-%m-%d')
    
    lista = double_list[1]
    korok = []
    
    for bd in lista: 
    
        try: 
            born = datetime.strptime(bd, '%Y-%m-%d')
            kor =  today-born
            korok.append(kor.days)
        except: 
            continue
        
    return np.average(korok)

def gyoztes_kor(df):
    
    if df["result"] == 1:
        return df["avg_home_age"]
    elif df["result"] == 2:
        return df["avg_away_age"]
    else:
        return pd.np.nan

def id_of_oldest_team_to_win_a_game(m_i_df):
    print(m_i_df.columns)
    print(m_i_df.head())
    print(m_i_df.loc[:,["result","avg_away_age","avg_home_age"]].apply(gyoztes_kor,axis=1))
    if m_i_df.loc[m_i_df.loc[:,["result","avg_away_age","avg_home_age"]].apply(gyoztes_kor,axis=1).idxmax(),"result"]==1:
        return str(m_i_df.loc[m_i_df.loc[:,["result","avg_away_age","avg_home_age"]].apply(gyoztes_kor,axis=1).idxmax(),"home_team"])
    else:
        return str(m_i_df.loc[m_i_df.loc[:,["result","avg_away_age","avg_home_age"]].apply(gyoztes_kor,axis=1).idxmax(),"away_team"])

def calculate_max_age(double_list):
    
    match = double_list[0]
    today = datetime.strptime(match, '%Y-%m-%d')
    
    lista = double_list[1]
    korok = []
    
    for bd in lista: 
    
        try: 
            born = datetime.strptime(bd, '%Y-%m-%d')
            kor =  today-born
            korok.append(kor.days)
        except: 
            continue
            
    try:
        maxkor=max(korok)
    except:
        maxkor=pd.np.nan
        
    return maxkor

def calculate_max_age_place(double_list):
    
    match = double_list[0]
    today = datetime.strptime(match, '%Y-%m-%d')
    
    lista = double_list[1]
    korok = []
    
    for bd in lista: 
    
        try: 
            born = datetime.strptime(bd, '%Y-%m-%d')
            kor =  today-born
            korok.append(kor.days)
        except: 
            korok.append(pd.np.nan)
            
    try:
        return [i for i, j in enumerate(korok) if j == max(korok)]
 
    except:
        return pd.np.nan
def try_height(x):
    
    try:
        magassag = int(current_app.p_i_dict[x]["height"])
        return magassag
        
    except:
        return pd.np.nan 

def avg_height(lista):
    
    
    # np.average(list(map(lambda x: int(p_i_dict[x]["height"]),lista)))
        
    return np.nanmean(list(map(try_height,lista)))

def height_diff_OC(df):
    
    if df["result"] == 1:
        return df["AVG_HEIGHT_A"]-df["AVG_HEIGHT_H"]
    elif df["result"] == 2:
        return df["AVG_HEIGHT_H"]-df["AVG_HEIGHT_A"]
    else:
        return 0

def stad_cap(x):
    
    try:
        return current_app.t_i_dict[x]["seats"]
    except:
        return pd.np.nan
def stad_diff_OC(df):
    
    if df["result"] == 1:
        return df["stad_cap_a"]-df["stad_cap_h"]
    elif df["result"] == 2:
        return df["stad_cap_h"]-df["stad_cap_a"]
    else:
        return 0

def height_to_list(x):
    
    lista = []
    for i in x:
        
        try: 
        
            asd = current_app.p_i_dict[i]["height"]
            lista.append(asd)
            
        except:
            continue 
        
    return lista

def try_height_2(x):
    
    try:
        #magassag = int(x)
        return int(x)
        
    except:
        return pd.np.nan 

def calculate_valamelyik_age(double_list):
    
    match = double_list[0]
    today = datetime.strptime(match, '%Y-%m-%d')
    
    lista = double_list[1]
    korok = []
    
    for bd in lista: 
    
        try: 
            born = datetime.strptime(bd, '%Y-%m-%d')
            kor =  today-born
            korok.append(kor.days)
        except: 
            korok.append(pd.np.nan)
            
    return korok
app.szar_lista = list(set(list(app.m_o_df["at"]) + list(app.m_o_df["ht"])))

app.eggyel_jobb_lista = []

for x in app.szar_lista:
    
    if  x[-1]=='\xa0':
        app.eggyel_jobb_lista.append(x[:-1])
    else:
        app.eggyel_jobb_lista.append(x)
        
app.eggyel_jobb_lista = list(set(app.eggyel_jobb_lista))


#############################################xx
app.fasza_nevek = list(app.t_i_df.name.values)
app.szar_nev_szotar = { i : "" for i in app.eggyel_jobb_lista }
##############################################
def nevek_re(x):
    if  x[-1]=='\xa0':
        return x[:-1]
    else:
        return x

        
##############################################
app.keywords = app.eggyel_jobb_lista
app.groups = app.fasza_nevek 

print("eddig ELJUT")
#print(app.groups)
#print(app.keywords )

app.assigned_groups = [max(app.groups, key=lambda g: Levenshtein.ratio(g, k))  for k in app.keywords]

app.df = pd.DataFrame({"Keyword": app.keywords, "Group": app.assigned_groups})

app.m_o_df["at"] = app.m_o_df["at"].apply(nevek_re)
app.m_o_df["ht"] = app.m_o_df["ht"].apply(nevek_re)

app.m_o_df = pd.merge(pd.merge(pd.merge(pd.merge(app.m_o_df, app.df, left_on='at', right_on='Keyword', how='left')\
    .drop(['Keyword'], axis=1)\
    ,app.t_i_df, left_on='Group', right_on='name', how='left')\
    .rename(columns={'Group':'Group_at','team_id': 'team_id_at', 'name': 'name_at','seats':'seats_at'})
    ,app.df, left_on='ht', right_on='Keyword', how='left')\
    .drop(['Keyword'], axis=1)\
    ,app.t_i_df, left_on='Group', right_on='name', how='left')\
    .rename(columns={"Group":"Group_ht",'team_id': 'team_id_ht', 'name': 'name_ht','seats':'seats_ht'})\
    .drop(['Group_ht',"Group_at"], axis=1)
########################################################
app.p_i_dict = app.p_i_df.set_index('playerid').T.to_dict('dict')


def eletkor_to_list(x):
    
    lista = []
    for i in x:
        
        try: 
        
            asd = current_app.p_i_dict[i]["dob"]
            lista.append(asd)
            
        except:
            continue 
        
    return lista
        
def calculate_age(double_list):
    
    match = double_list[0]
    today = datetime.strptime(match, '%Y-%m-%d')
    
    lista = double_list[1]
    korok = []
    
    for bd in lista: 
    
        try: 
            born = datetime.strptime(bd, '%Y-%m-%d')
            kor =  today-born
            korok.append(kor.days)
        except: 
            continue
        
    return np.average(korok)

def oddsx_to_float(x):
    try:
        return float(x)
    except:
        return pd.np.nan

def fav_age(df):
    
    if df["odds1"]>df["odds2"]:
        return df["avg_away_age"]
    elif df["odds1"]<df["odds2"]:
        return df["avg_home_age"]
    else:
        return pd.np.nan

def fav_age(df):
    
    if df["odds1"]>df["odds2"]:
        return df["avg_away_age"]
    elif df["odds1"]<df["odds2"]:
        return df["avg_home_age"]
    else:
        return pd.np.nan

def udog_age(df):
    
    if df["odds1"]<df["odds2"]:
        return df["avg_away_age"]
    elif df["odds1"]>df["odds2"]:
        return df["avg_home_age"]
    else:
        return pd.np.nan

def win_udog_h(df):
    
    if df["odds1"]>df["odds2"] and df["result"]==1.0:
        return 1
    else:
        return 
    
def win_udog_a(df):
    
    if df["odds1"]<df["odds2"] and df["result"]==2.0:
        return 1
    else:
        return 0

def lose_fav_h(df):
    
    if df["odds1"]>df["odds2"] and df["result"]==2.0:
        return 1
    else:
        return 0
    
def lose_fav_a(df):
    
    if df["odds1"]<df["odds2"] and df["result"]==1.0:
        return 1
    else:
        return 0

def looser(df):
    
    if df["result"]==1.0:
        return df['team_id_at']
    elif df["result"]==2.0:
        return  df['team_id_ht']
    else:
        return pd.np.nan
    
def looser_odds(df):
    
    if df["result"]==1.0:
        return df['odds1']
    elif df["result"]==2.0:
        return  df['odds2']
    else:
        return pd.np.nan

def jatekosok(df):
    
    current_app.p_v_dict[df["playerid"]]["dicts"]["date"].append(df["date"])
    current_app.p_v_dict[df["playerid"]]["dicts"]["y"].append(df["y"])

def player_value(df):
    
    values = []
    
    try: 
        lista = df["home_players"]
        
    except:
        lista = df["away_players"]
        
    
    for i in lista:
    
        try:
            
            if df["date"] in current_app.p_v_dict[i]["dicts"]["date"]:
                
                values.append(current_app.p_v_dict[i]["dicts"]["y"][np.searchsorted(current_app.p_v_dict[i]["dicts"]["date"],df["date"])])
            else: 
                values.append(current_app.p_v_dict[i]["dicts"]["y"][np.searchsorted(current_app.p_v_dict[i]["dicts"]["date"],df["date"])-1])
                
                
        except:
            values.append(pd.np.nan)
    
    return values

def values_diff(df):
    
    return abs(sum(df["away_values"])-sum(df["home_values"]))

def values_diff_ups(df):
    
    if df["result"]==1:
        return sum(df["home_values"])-sum(df["away_values"])
    elif df["result"]==2:
        return sum(df["away_values"])-sum(df["home_values"])
    else:
        return 0
def values_diff_ups(df):
    
    if df["result"]==1:
        return sum(df["home_values"])-sum(df["away_values"])
    elif df["result"]==2:
        return sum(df["away_values"])-sum(df["home_values"])
    else:
        return 0

def values_diff_ups_odds(df):
    
    if df["odds1"]> df["odds2"]:
        return sum(df["home_values"])-sum(df["away_values"])
    elif df["odds1"]< df["odds2"]:
        return sum(df["away_values"])-sum(df["home_values"])
    else:
        return 0
        

        

    



if __name__ == '__main__':
    app.run(debug=True,port=5112)
