import pandas as pd
import json

args = json.load(open('data.json','r'))

df = pd.read_csv(args['datafile'])

df=df.drop_duplicates()
#df=df.sort_values(by='price')


df1=df.loc[df['stars']==0]
df2=df.loc[df['stars']==0.5]
df3=df.loc[df['stars']==1]
df4=df.loc[df['stars']==1.5]
df5=df.loc[df['stars']==2]
df6=df.loc[df['stars']==2.5]
df7=df.loc[df['stars']==3]
df8=df.loc[df['stars']==3.5]
df9=df.loc[df['stars']==4]
df10=df.loc[df['stars']==4.5]
df11=df.loc[df['stars']==5]


df1.to_pickle(args['staging_folder'] + '/filtered0.pkl')
df2.to_pickle(args['staging_folder'] + '/filtered5.pkl')
df3.to_pickle(args['staging_folder'] + '/filtered10.pkl')
df4.to_pickle(args['staging_folder'] + '/filtered15.pkl')
df5.to_pickle(args['staging_folder'] + '/filtered20.pkl')
df6.to_pickle(args['staging_folder'] + '/filtered25.pkl')
df7.to_pickle(args['staging_folder'] + '/filtered30.pkl')
df8.to_pickle(args['staging_folder'] + '/filtered35.pkl')
df9.to_pickle(args['staging_folder'] + '/filtered40.pkl')
df10.to_pickle(args['staging_folder'] + '/filtered45.pkl')
df11.to_pickle(args['staging_folder'] + '/filtered50.pkl')
