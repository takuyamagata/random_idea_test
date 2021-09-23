import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os, argparse, glob, pickle

from bsonhousehold.house import HouseDay
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%%
def humidity_latent_event_detection(x):
    """
        Max-sum (Viterbi) algorithm 
    """
    k1 = 0.1   # true value 0.25
    k2 = 0.5    # true value 0.5

    bg_percentile = 10
    bg_window_len = 240//3

    # transition probability s1->s2 = T[s1,s2]
    T = np.array([[0.8, 0.2], 
                  [0.2, 0.8]]) 

    N = len(x)-1

    s = np.zeros((N, 2))
    s_in_x = np.zeros((N, 2))
    s_sel = np.zeros((N, 2))

    # compute the background humidity level
    x_bg = np.zeros(N+1)
    x_bg[0] = x[0]
    for n in range(N):
        if n < bg_window_len:
            idx = np.arange(0,n+1)
        else:
            idx = np.arange(n-bg_window_len+1,n+1)
        x_bg[n+1] = np.percentile(x[idx], bg_percentile)

    # prepare messages from x[n] and x[n+1] 
    for n in range(N):
        for s_ in [0, 1]:
            x_ = x[n] + (1-s_)*k1*(x_bg[n] - x[n]) + s_*k2*(100. - x[n])
            s_in_x[n, s_] = np.log(norm.pdf(x[n+1], loc=x_, scale=5))
    
    # message passing to root node (s[0]) with max-sum (Viterbi)
    s_ = np.zeros(2)
    s_in_ = np.zeros(2) # message from the future s[n]
    for n in reversed(range(N)):
        s_[0] = s_in_[0] + s_in_x[n,0]
        s_[1] = s_in_[1] + s_in_x[n,1]
        s_in_[0] = np.max([T[0,0]+s_[0], T[0,1]+s_[1]])
        s_in_[1] = np.max([T[1,0]+s_[0], T[1,1]+s_[1]])
        s_sel[n,0] = np.argmax([T[0,0]+s_[0], T[0,1]+s_[1]])
        s_sel[n,1] = np.argmax([T[1,0]+s_[0], T[1,1]+s_[1]])
        
    # decode (trace the selected path)
    s = np.zeros(N)
    s[0] = 0
    for n in range(N-1):
        s[n+1] = s_sel[n+1,int(s[n])]

    return s

def get_room_uid(hcat, room='bathroom'):
    """
    Returns UID for the given room
    """
    uid = []
    if 'house_location' in hcat.keys():
        for uid_, location_, description_ in zip(hcat['href'], hcat['house_location'], hcat['description']):
            if (location_ is not None) and (room in location_) and ('Enviro' in description_):
                uid.append(uid_)
                print(f"found '{room}' in hcat data: {uid_}")
        
    if len(uid) == 0:
        # couldn't find the room, let user select one.
        for n, uid_ in enumerate(env_dict['HDC_HUM'].keys()):
            print(f'{n}: {uid_}')
        idx = input(f'select one uid 0-{n}')
        uid = list(env_dict['HDC_HUM'].keys())[int(idx)]

    return uid
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyse_env_data')
    parser.add_argument('-d', '--datasets-path', dest='datasets_path',
                        type=str,
                        default='/export/sphere/rodan/datasets/100-homes',
                        help='Path to the sphere mongodb folders')
    parser.add_argument('-o', '--output-path', dest='output_path', type=str,
                        default='/export/sphere/rodan/datasets/ty17095/',
                        help='Path to export all results')
    parser.add_argument('-hid', '--house-id', dest='hid', type=str,
                        default='9999',
                        help='house id') #4510, 9665
    parser.add_argument('-r', '--room', dest='room', type=str,
                        default='bathroom',
                        help='room name') # 'bathroom', 'kitchen'
    config = parser.parse_args()

    day_id_list = glob.glob(os.path.join(config.datasets_path, config.hid, '????-??-??/'))

    env_dict = {}
    for day_id_path in day_id_list:
        split_ = day_id_path.split(os.sep)
        day_id = split_[-2]
        print(day_id)
        folder = os.path.join(config.datasets_path, config.hid, day_id)
        hd = HouseDay(config.hid, folder, day_id)
        env = hd.get_env(['HDC_HUM'])  # get humidity data
        hcat = hd.get_hypercat()       # get hypercat data

        uid = get_room_uid(hcat, room=config.room) # get uid for the target room
        df = env['HDC_HUM'][uid] # get data for uid

        # re-sample data
        df = df.resample('3min').mean()
        df = df.resample('3min').interpolate()

        # detect the latent states
        x = df['value'].to_numpy()
        s = humidity_latent_event_detection(x)
        df['state'] = np.append(s,0)
        df.to_csv(os.path.join(config.output_path, f'hum_state_{config.hid}_{day_id}.csv'))

        # plot results
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df.index, y=df['value'], name="humidity"),
                    secondary_y=False,
                    )
        fig.add_trace(go.Scatter(x=df.index, y=df['state'], name="state"),
                    secondary_y=True,
                    )
        fig.update_xaxes(title="date-time") # X軸タイトルを指定
        fig.update_yaxes(title="humidity [%]",
                        secondary_y=False) # Y軸タイトルを指定
        fig.update_yaxes(title="State",
                        secondary_y=True) # Y軸タイトルを指定
        fig.update_xaxes(rangeslider={"visible":True}) # X軸に range slider を表示（下図参照）
        fig.update_layout(title=f"Humidity for {config.hid} {day_id}") # グラフタイトルを設定
        fig.write_html(os.path.join(config.output_path, f'hum_state_{config.hid}_{day_id}.html'))
