import argparse
import glob
import os
import random
import re

import numpy as np
from scipy import ndimage

# import wfdb

# set numpy seed for reproduction
SEED = 3010
np.random.seed(SEED)

# PARENT_DIR = path.abspath(path.join("../../.."))
FRAME_RATE = 200  # was 250
M = 201  # size of snippet windows
U = [10, 1]  # overlap of snippets in samples ==> index{0} = train, index{1} = test
NO_SIGNALS = 4  # choose the number of signals extracted from the merged data set; relates to the "_pick" parameter

# optional parameter to pick a specific signal 0-3 in order to train and test the corresponding cnn model
# in the context of the voting approach; default value = 0 ==> first signal (ecg)
# signals are stored in the "merged" directory in one data set with K channels, where K indicates the number of signals
# 0=ecg, 1=bcg, 2=ppg, 3=ippg
_PICK = 0


def signal_selection(no_sig, sel=0):
    if no_sig == 1:
        channels = [sel]
    elif no_sig == 2:
        channels = [1, 3]
    elif no_sig == 3:
        channels = [0, 2, 3]
    else:
        channels = [i for i in range(no_sig)]
    return channels


def check_processed_data(x, y):
    print("=> checking processed data.")
    length_x = x.shape[0]
    length_y = len(y)
    assert length_x == length_y
    print(f"==> len(x): {length_x} == len(y): {length_y}")

    beats, non_beats = 0, 0
    for _y in y:
        if _y == 1:
            beats += 1
        else:
            non_beats += 1
    assert len(x[random.randint(0, length_x - 1), :]) == 201
    print("==> segment length checked.")
    return beats, non_beats


def maintain_class_balance(segment_set, label_set):
    all_zero_index = np.where(label_set == 0)[0].tolist()
    all_one_index = np.where(label_set == 1)[0].tolist()
    if len(all_zero_index) >= len(all_one_index):
        del_no = len(all_zero_index) - len(all_one_index)
        np.random.shuffle(all_zero_index)
        to_delete = all_zero_index[:del_no]
        print(f"==> removing #{del_no} zero-labeled snippets.")
    elif len(all_one_index) >= len(all_zero_index):
        del_no = len(all_one_index) - len(all_zero_index)
        np.random.shuffle(all_one_index)
        to_delete = all_one_index[:del_no]
        print(f"==> removing #{del_no} one-labeled snippets.")
    else:
        to_delete = []
    segment_set = np.delete(segment_set, to_delete, axis=0)
    label_set = np.delete(label_set, to_delete, axis=0)
    return segment_set, label_set


def define_location_pulses(actual_annotations):
    all_location_pulses = []
    for annotation in actual_annotations:
        left = annotation - 37
        right = annotation + 38
        if left >= 0:
            single_location_pulse = list(range(left, right))
            all_location_pulses.append(single_location_pulse)
    return all_location_pulses


def signal_improvement(inc_signal):
    win_size = int(0.2 * FRAME_RATE)
    med_flt_sig = ndimage.median_filter(inc_signal, win_size)
    win_size = int(0.6 * FRAME_RATE)
    med_flt_sig = ndimage.median_filter(med_flt_sig, win_size)
    flt_sig = inc_signal - med_flt_sig
    # norm_sig = minmax_scale(flt_sig, feature_range=(-1, 1))
    # return norm_sig
    return flt_sig


def process_data(data, ant_data, u, trained_data=None, test=False):
    x_all_dict = {}
    y_all_dict = {}

    for i in range(len(data)):
        data_file = data[i]
        scenario = re.search(r"_(City|Highway|Rural)\.npy$", data_file).group(1)
        subject = re.search(r"mergeData_(\d{4})_", data_file).group(1)

        ant_file = ant_data[i]
        signal_data = np.load(data_file)
        annotations = np.load(ant_file)

        channels = signal_selection(NO_SIGNALS, _PICK)
        channels = tuple(channels)
        signal_data = signal_data[:, channels]

        if np.sum(signal_data) == 0:
            print(f"=> no signal data available (skipping {data_file}).")
            continue

        for channel in range(NO_SIGNALS):
            signal_data[:, channel] = signal_improvement(signal_data[:, channel])

        location_pulses = define_location_pulses(annotations)
        num_rows = int((len(signal_data) - M) / u)
        x = np.zeros((num_rows, M, NO_SIGNALS), dtype=np.float64)
        y = np.zeros((num_rows, 1), dtype=np.float64)

        left = 0
        right = M
        beats = 0
        for ct in range(num_rows):
            current_segment = signal_data[left:right]
            current_segment_in_samples = list(range(left, right))
            if len(current_segment) == M:
                x[ct, :] = current_segment
                current_segment_mid_sample = current_segment_in_samples[int(len(current_segment_in_samples) / 2)]
                is_beat = any(current_segment_mid_sample in location_pulse for location_pulse in location_pulses)
                if is_beat:
                    y[ct, :] = 1
                    beats += 1
                else:
                    y[ct, :] = 0
                left += u
                right = left + M

        new_x, new_y = maintain_class_balance(x, y)
        beats, non_beats = check_processed_data(new_x, new_y)

        if subject not in x_all_dict:
            x_all_dict[subject] = {}
            y_all_dict[subject] = {}
        if scenario not in x_all_dict[subject]:
            x_all_dict[subject][scenario] = new_x
            y_all_dict[subject][scenario] = new_y
        else:
            x_all_dict[subject][scenario] = np.append(x_all_dict[subject][scenario], new_x, axis=0)
            y_all_dict[subject][scenario] = np.append(y_all_dict[subject][scenario], new_y, axis=0)

    return x_all_dict, y_all_dict


def start(signal_data, annotation_data, rtype, train=False, test=False):
    signal_data = sorted(signal_data)
    annotation_data = sorted(annotation_data)
    assert len(signal_data) == len(annotation_data)
    test_data = np.array(signal_data)
    test_annotation = np.array(annotation_data)
    training_data = np.array(signal_data)
    training_annotation = np.array(annotation_data)
    train_data_names = [re.findall(r"[+-]?\d+", fn)[0] for fn in training_data]

    print("\n#----- GENERAL META DATA -----#")
    print(
        f"TRAIN FILES: {train_data_names}\nRECORD TYPE: {rtype}\nNO SIGNALS: {NO_SIGNALS}"
        f"\nSEGMENT LENGTH: {M}\nOVERLAP TRAIN: {U[0]}\nOVERLAP TEST: {U[1]}"
    )

    if train:
        print("\n#----- GENERATING TRAIN DATA -----#")
        x_all_dict, y_all_dict = process_data(training_data, training_annotation, U[0])
        for subject, scenarios in x_all_dict.items():
            for scenario, x_data in scenarios.items():
                y_data = y_all_dict[subject][scenario]
                np.save(
                    f"data/preprocessed/train/{rtype}/{subject}_{scenario}_x.npy",
                    x_data,
                )
                np.save(
                    f"data/preprocessed/train/{rtype}/{subject}_{scenario}_y.npy",
                    y_data,
                )

    if test:
        print("\n#----- GENERATING TEST DATA -----#")
        x_all_dict, y_all_dict = process_data(test_data, test_annotation, U[1], train_data_names, True)
        for subject, scenarios in x_all_dict.items():
            for scenario, x_data in scenarios.items():
                y_data = y_all_dict[subject][scenario]
                np.save(
                    f"data/preprocessed/test/{rtype}/{subject}_{scenario}_x.npy",
                    x_data,
                )
                np.save(
                    f"data/preprocessed/test/{rtype}/{subject}_{scenario}_y.npy",
                    y_data,
                )

    print("> DONE.")


def run(r_type, train, test):
    print("> INIT.")
    local_data_dir = f"data/merged/{r_type}/"
    local_ant_dir = f"data/annotations/{r_type}/"
    ext = "*.npy"

    print(local_data_dir)
    all_data_files = glob.glob(local_data_dir + ext)
    all_ant_files = glob.glob(local_ant_dir + ext)
    print("> FILES COLLECTED.")
    print(all_data_files)

    os.makedirs(f"data/preprocessed/train/{r_type}/", exist_ok=True)
    os.makedirs(f"data/preprocessed/test/{r_type}/", exist_ok=True)

    start(all_data_files, all_ant_files, r_type, train, test)


def argument_parser():
    parser = argparse.ArgumentParser(description="This script is for generating test and train data.")
    parser.add_argument("record_type", type=str, help='Choose "motion" or "rest"')
    parser.add_argument("--train", type=int)
    parser.add_argument("--test", type=int)
    params = parser.parse_args(["motion", "--train", "1", "--test", "1"])
    return params


if __name__ == "__main__":
    p = argument_parser()
    run(p.record_type, p.train, p.test)
