"""
Preprocess PAMAP2 dataset
Download from: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
Usage: unzip and put raw data in 'datasets/pamap2/raw'; run this script

Hard facts:
9 subjects
18 activities:
    - lying
    - sitting
    - standing
    - walking
    - running
    - cycling
    - Nordic walking
    - watching TV
    - computer work
    - car driving
    - ascending stairs
    - descending stairs
    - vacuum cleaning
    - ironing
    - folding laundry
    - house cleaning
    - playing soccer
    - rope jumping
Sampling rate: 100Hz
Sensitivity: +- 16g
Sensor placement: Wrist, Chest, Ankle
"""

import argparse
from scipy import interpolate
import os
from tqdm import tqdm
import numpy as np
import glob
import pickle


def get_data_content(data_path):
    """
    Loads the data from the raw PAMAP2 file and returns the wrist, chest and ankle data.
    
    Parameters:
        data_path (str): path to the raw data file
        
    Returns:
        wristDatContent (np.array): wrist data
        chestDatContent (np.array): chest data
        ankleDatContent (np.array): ankle data
    """
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)
    timestamp_idx = 0
    label_idx = 1
    wrist_x_idx, wrist_y_idx, wrist_z_idx = 4, 5, 6
    chest_x_idx, chest_y_idx, chest_z_idx = 21, 22, 23
    ankle_x_idx, ankle_y_idx, ankle_z_idx = 38, 39, 40
    # print value counts of labels
    wristDatContent = datContent[:, [timestamp_idx, label_idx, wrist_x_idx, wrist_y_idx, wrist_z_idx]]
    wristDatContent = wristDatContent.astype(np.float64)
    # interpolate missing values in wrist data
    for i in range(2, 5):
        x_orig = wristDatContent[:, 0]
        y_orig = wristDatContent[:, i]
        x = x_orig[~np.isnan(y_orig)]
        y = y_orig[~np.isnan(y_orig)]
        f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        ynew = f(x_orig)
        wristDatContent[:, i] = ynew

    chestDatContent = datContent[:, [timestamp_idx, label_idx, chest_x_idx, chest_y_idx, chest_z_idx]]
    chestDatContent = chestDatContent.astype(np.float64)
    # interpolate missing values in chest data
    for i in range(2, 5):
        x_orig = chestDatContent[:, 0]
        y_orig = chestDatContent[:, i]
        x = x_orig[~np.isnan(y_orig)]
        y = y_orig[~np.isnan(y_orig)]
        f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        ynew = f(x_orig)
        chestDatContent[:, i] = ynew
    
    ankleDatContent = datContent[:, [timestamp_idx, label_idx, ankle_x_idx, ankle_y_idx, ankle_z_idx]]
    ankleDatContent = ankleDatContent.astype(np.float64)
    # interpolate missing values in ankle data
    for i in range(2, 5):
        x_orig = ankleDatContent[:, 0]
        y_orig = ankleDatContent[:, i]
        x = x_orig[~np.isnan(y_orig)]
        y = y_orig[~np.isnan(y_orig)]
        f = interpolate.interp1d(x, y, kind='linear', fill_value='extrapolate')
        ynew = f(x_orig)
        ankleDatContent[:, i] = ynew
        
    return wristDatContent, chestDatContent, ankleDatContent


def process_all(file_paths, X_wrist_path, X_chest_path, X_ankle_path, y_path, pid_path):
    """
    Processes all the data files and saves the wrist, chest and ankle data to numpy files.
    
    Parameters:
        file_paths (list): list of paths to the raw data files
        X_wrist_path (str): path to save wrist data
        X_chest_path (str): path to save chest data
        X_ankle_path (str): path to save ankle data
        y_path (str): path to save labels
        pid_path (str): path to save participant ids
    """
    wrist_X, chest_X, ankle_X = [], [], []
    y = []
    pid = []
    user_list = []

    for file_path in tqdm(file_paths):
        subject_id = int(file_path.split("/")[-1][-5])

        user = f"S{subject_id}"
        if user not in user_list:
            user_list.append(user)

        wristDatContent, chestDatContent, ankleDatContent = get_data_content(file_path)
        
        current_wrist_X, current_chest_X, current_ankle_X = \
            wristDatContent[:, 2:5], chestDatContent[:, 2:5], ankleDatContent[:, 2:5]
        current_y = chestDatContent[:, 1]
        ids = np.full(shape=len(current_y), fill_value=subject_id, dtype=np.int64)
        if len(wrist_X) == 0:
            wrist_X = current_wrist_X
            chest_X = current_chest_X
            ankle_X = current_ankle_X
            y = current_y
            pid = ids
        else:
            wrist_X = np.concatenate([wrist_X, current_wrist_X])
            chest_X = np.concatenate([chest_X, current_chest_X])
            ankle_X = np.concatenate([ankle_X, current_ankle_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    y = y.flatten()

    # discard transient activities class
    null_filter = y != 0
    pid = pid[null_filter]
    wrist_X = wrist_X[null_filter]
    chest_X = chest_X[null_filter]
    ankle_X = ankle_X[null_filter]
    y = y[null_filter]

    print(f"Flattened shape: {wrist_X.shape}")

    window_size = 200
    channels = 6

    pid = np.array(pid)
    wrist_X = np.array(wrist_X)
    chest_X = np.array(chest_X)
    ankle_X = np.array(ankle_X)
    y = np.array(y)
    
    # replace y values with actual class names using dictionary
    y_dict = {
        0: "lying",
        1: "sitting",
        2: "standing",
        3: "walking",
        4: "running",
        5: "cycling",
        6: "nordic_walking",
        7: "ascending_stairs",
        8: "descending_stairs",
        9: "vacuum_cleaning",
        10: "ironing",
        11: "rope_jumping",
    }

    # filter out activities that are not relevant
    mask = np.isin(y, list(y_dict.keys()))
    pid = pid[mask]
    wrist_X = wrist_X[mask]
    chest_X = chest_X[mask]
    ankle_X = ankle_X[mask]
    y = y[mask]

    
    y_string = np.array([y_dict[i] for i in y])

    device_list = ['hand', 'chest', 'ankle']
    session_list = [y_dict[key] for key in y_dict.keys()]
    info = {
        'device_list': device_list,
        'user_list': user_list,
        'session_list': session_list
    }
    training = dict()
    testing = dict()
    hand_data = dict()
    chest_data = dict()
    ankle_data = dict()

    for user in user_list:
        hand_data[user] = dict()
        chest_data[user] = dict()
        ankle_data[user] = dict()

    for device in device_list:
        for user in user_list:
            uid = int(user[-1])
            usr_labels = y[pid == uid]

            print(f"User labels shape: {usr_labels.shape}")

            for session in session_list:
                sessionid = list(y_dict.keys())[list(y_dict.values()).index(session)]
                session_mask = np.isin(usr_labels, [sessionid])
                session_labels = usr_labels[session_mask]

                print(f"Session labels shape: {session_labels.shape}")


                if device == 'hand':
                   device_data = hand_data
                   device_X = wrist_X
                elif device == 'chest':
                   device_data = chest_data
                   device_X = chest_X
                elif device == 'ankle':
                   device_data = ankle_data
                   device_X = ankle_X

                data_u = device_X[pid == uid]
                data_u_s = data_u[session_mask]

                print(f"Shape {device} {user} {session}: {data_u_s.shape}")

                data_u_s = data_u_s[:((data_u_s.shape[0]//(window_size*channels))*(window_size*channels))]
                newshape = ((data_u_s.shape[0]*data_u_s.shape[1])//(window_size*channels), window_size, channels)
                data_u_s = np.reshape(data_u_s, newshape=newshape)
                device_data[user][session] = [data_u_s, session_labels[:data_u_s.shape[0]]]

    training = {
        'hand': hand_data,
        'chest': chest_data,
        'ankle': ankle_data
    }

    testing = {
        'hand': hand_data,
        'chest': chest_data,
        'ankle': ankle_data
    }

    dataset_full = (info, training, testing)

    with open('pamap2_test.dat', 'wb') as f:
        pickle.dump(dataset_full, f)

    """np.save(X_wrist_path, wrist_X)
    np.save(X_chest_path, chest_X)
    np.save(X_ankle_path, ankle_X)
    np.save(y_path, y_string)
    np.save(pid_path, pid)"""



def get_write_paths(data_root):
    """
    Creates paths to save the wrist, chest and ankle data.
    
    Parameters:
        data_root (str): path to the root directory
    
    Returns:
        X_wrist_path (str): path to save wrist data
        X_chest_path (str): path to save chest data
        X_ankle_path (str): path to save ankle data
        y_path (str): path to save labels
        pid_path (str): path to save participant ids
    """
    X_wrist_path = os.path.join(data_root, "X_wrist.npy")
    X_chest_path = os.path.join(data_root, "X_chest.npy")
    X_ankle_path = os.path.join(data_root, "X_ankle.npy")
    y_path = os.path.join(data_root, "Y.npy")
    pid_path = os.path.join(data_root, "pid.npy")
    return X_wrist_path, X_chest_path, X_ankle_path, y_path, pid_path

def main(args):
    data_root = args.data_root
    output_root = args.output_root
    data_path = data_root + "/Protocol/"
    protocol_file_paths = glob.glob(os.path.join(data_path, "*.dat"))
    data_path = data_root + "/Optional/"
    optional_file_paths = glob.glob(os.path.join(data_path, "*.dat"))
    file_paths = protocol_file_paths #+ optional_file_paths
    
    X_wrist_path, X_chest_path, X_ankle_path, y_path, pid_path = get_write_paths(output_root)
    process_all(file_paths, X_wrist_path, X_chest_path, X_ankle_path, y_path, pid_path)
    print("Saved files to ", output_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess PAMAP2 dataset')
    parser.add_argument('--data_root', type=str, default="datasets/pamap2/raw", help='Path to the root directory')
    parser.add_argument('--output_root', type=str, default="datasets/pamap2", help='Path to the output directory')
    args = parser.parse_args()
    main(args)
