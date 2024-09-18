import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)

import numpy as np 
import streamlit as st
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math 
from tqdm import tqdm

# bsoid_app/bsoid_utilities/likelihoodprocessing.py
def adp_filt(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    currdf = np.array(currdf[1:])
    for header in pose:
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    curr_df1 = currdf[:, 1:]
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[:, x].astype(float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(float)
    return currdf_filt, perc_rect

# bsoid_app/bsoid_utilities/likelihoodprocessing.py
def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg

# bsoid_app/extract_features.py # edited
#def compute(processed_input_data, file_j_df_array, framerate):

def compute(processed_input_data, framerate=30):
        #if st.button("__Extract Features__"):
        #    funfacts = randfacts.getFact()
        #    st.info(str.join('', ('Extracting... Here is a random fact: ', funfacts)))
        #try:
        #    [features, scaled_features] = load_feats(working_dir, prefix)
        #except:
        window = int(np.round(0.05 / (1 / framerate)) * 2 - 1)
        f = []
        my_bar = st.progress(0)
        for n in range(len(processed_input_data)):
            data_n_len = len(processed_input_data[n])
            dxy_list = []
            disp_list = []

            # (MB, 5/21/2024): calculate the interframe distance of the body parts from frame n to frame n + 1 

            for r in range(data_n_len):
                if r < data_n_len - 1:
                    disp = []
                    for c in range(0, processed_input_data[n].shape[1], 2):
                        
                        disp.append(
                            np.linalg.norm(processed_input_data[n][r + 1, c:c + 2] -
                                        processed_input_data[n][r, c:c + 2]))
                    disp_list.append(disp) # (MB, 5/21/2024): append the interframe distances for each row to the big list
                
                # (MB, 5/21/2024): calculate the pairwise distance vectors from each body part to each of the other body parts

                dxy = []
                for i, j in itertools.combinations(range(0, processed_input_data[n].shape[1], 2), 2):
                    dxy.append(processed_input_data[n][r, i:i + 2] -
                            processed_input_data[n][r, j:j + 2])
                
                dxy_list.append(dxy) # (MB, 5/21/2024): append the pairwise distance vectors for each row to the big list
            disp_r = np.array(disp_list)
            dxy_r = np.array(dxy_list)
            disp_boxcar = []
            dxy_eu = np.zeros([data_n_len, dxy_r.shape[1]])
            ang = np.zeros([data_n_len - 1, dxy_r.shape[1]])
            dxy_boxcar = []
            ang_boxcar = []

            # (MB, 5/21/2024): average the interframe distance values with a rolling average with a window size of 3 ( window = int(np.round(0.05 / (1 / framerate)) * 2 - 1) )

            for l in range(disp_r.shape[1]):
                disp_boxcar.append(boxcar_center(disp_r[:, l], window))
            
            for k in range(dxy_r.shape[1]): # (MB, 5/21/2024): for each column of dxr
                for kk in range(data_n_len): # (MB, 5/21/2024): for each row
                    dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :]) # (MB, 5/21/2024): calculate the pairwise euclidean distance
                    if kk < data_n_len - 1:
                        
                        # (MB, 5/21/2024): calculate the interframe angles of the body parts 
                        b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                        a_3d = np.hstack([dxy_r[kk, k, :], 0])
                        c = np.cross(b_3d, a_3d)
                        ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                            math.atan2(np.linalg.norm(c),
                                                    np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
                dxy_boxcar.append(boxcar_center(dxy_eu[:, k], window))
                ang_boxcar.append(boxcar_center(ang[:, k], window))
            disp_feat = np.array(disp_boxcar)
            dxy_feat = np.array(dxy_boxcar)
            ang_feat = np.array(ang_boxcar)
            f.append(np.vstack((dxy_feat[:, 1:], ang_feat, disp_feat)))
            my_bar.progress(round((n + 1) / len(processed_input_data) * 100))
        
        #frame_mapping  = []
        #frame_number = []

        # (MB, 5/21/2024): calculate the mean of the distance features and the sum of the angle features in a window size of 3

        for m in range(0, len(f)):
            f_integrated = np.zeros(len(processed_input_data[m]))
            for k in range(round(framerate / 10), len(f[m][0]), round(framerate / 10)):
                #frame_number.append(k-1)
                #frame_mapping.append(file_j_df_array[k-1+2][0])
                if k > round(framerate / 10):
                    f_integrated = np.concatenate(
                        (f_integrated.reshape(f_integrated.shape[0], f_integrated.shape[1]),
                        np.hstack((np.mean((f[m][0:dxy_feat.shape[0],
                                            range(k - round(framerate / 10), k)]), axis=1),
                                    np.sum((f[m][dxy_feat.shape[0]:f[m].shape[0],
                                            range(k - round(framerate / 10), k)]), axis=1)
                                    )).reshape(len(f[0]), 1)), axis=1
                    )
                else:
                    f_integrated = np.hstack(
                        (np.mean((f[m][0:dxy_feat.shape[0], range(k - round(framerate / 10), k)]), axis=1),
                        np.sum((f[m][dxy_feat.shape[0]:f[m].shape[0],
                                range(k - round(framerate / 10), k)]), axis=1))).reshape(len(f[0]), 1)
            if m > 0:
                features = np.concatenate((features, f_integrated), axis=1)
                scaler = StandardScaler()
                scaler.fit(f_integrated.T)
                scaled_f_integrated = scaler.transform(f_integrated.T).T
                scaled_features = np.concatenate((scaled_features, scaled_f_integrated), axis=1)
            else:
                features = f_integrated
                scaler = StandardScaler()
                scaler.fit(f_integrated.T)
                scaled_f_integrated = scaler.transform(f_integrated.T).T
                scaled_features = scaled_f_integrated
        features = np.array(features)
        scaled_features = np.array(scaled_features)

        #frame_mapping = np.array(frame_mapping)
        #frame_number = np.array(frame_number)

        return scaled_features, features#, frame_mapping, frame_number