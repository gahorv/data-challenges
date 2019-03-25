# coding: utf-8

import pandas as pd
import numpy as np
import json

input_json = json.load(open('input.json','r'))
data_args = json.load(open('data.json','r'))
data = pd.read_csv(data_args['staging_folder'] + '/filtered.csv')
num = 100 / len(data)
out = [] 

for place in input_json:

    r = 0
    by_lat = pd.DataFrame(np.zeros(shape = (0, 4))).rename(columns = {
        0: "lon", 1: "lat", 2: "name", 3: "distance"})
    by_lon = pd.DataFrame(np.zeros(shape = (0, 4))).rename(columns = {
        0: "lon", 1: "lat", 2: "name", 3: "distance"})

    while True:

        r += num

        if not by_lat.empty:
            lower_lat = by_lat.loc[0]["lat"]
            upper_lat = by_lat.loc[len(by_lat) - 1]["lat"]
            new_lat = data.loc[(abs(data["lat"] - place["lat"]) <= r) & ((data["lat"] < lower_lat) | \
                                    (data["lat"] > upper_lat))]
            new_lat["distance"] = (place['lon'] - new_lat['lon']) ** 2 + (place['lat'] - new_lat['lat']) ** 2
            by_lat = pd.concat([by_lat, new_lat], ignore_index = True).sort_values("lat").reset_index(drop = True)

        else:
            by_lat = pd.concat([by_lat, data.loc[abs(data["lat"] - place["lat"]) <= r].sort_values("lat").reset_index(
                drop = True)])
            by_lat["distance"] = (place['lon'] - by_lat['lon']) ** 2 + (place['lat'] - by_lat['lat']) ** 2

        if not by_lon.empty:
            lower_lon = by_lon.loc[0]["lon"]
            upper_lon = by_lon.loc[len(by_lon) - 1]["lon"]
            new_lon = data.loc[(abs(data["lon"] - place["lon"]) <= r) & ((data["lon"] < lower_lon) | \
                                    (data["lon"] > upper_lon))]
            new_lon["distance"] = (place['lon'] - new_lon['lon']) ** 2 + (place['lat'] - new_lon['lat']) ** 2
            by_lon = pd.concat([by_lon, new_lon], ignore_index = True).sort_values("lon").reset_index(drop = True)

        else:
            by_lon = pd.concat([by_lon, data.loc[abs(data["lon"] - place["lon"]) <= r].sort_values("lon").reset_index(
                drop = True)])
            by_lon["distance"] = (place['lon'] - by_lon['lon']) ** 2 + (place['lat'] - by_lon['lat']) ** 2        

        if by_lon.empty or by_lat.empty:
            continue

        lat_min = by_lat.iloc[[by_lat["distance"].idxmin()]].reset_index(drop = True)
        lon_min = by_lon.iloc[[by_lon["distance"].idxmin()]].reset_index(drop = True)

        if lat_min.equals(lon_min):
            closest_place = {'lon': lat_min.loc[0, 'lon'], 'lat': lat_min.loc[0, 'lat'],
                     'name': lat_min.loc[0, "name"]}
            out.append(closest_place.copy())
            break

json.dump(out,open('output.json','w'))
