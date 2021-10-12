from model_downloader import download_pretrained_model
from violin_fingering_model import violin_fingering_model
import numpy as np
import pretty_midi
import os
from tests import static
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_csv(csv_file_path):
    beat_type_dict = {'': 0, '1th': 1, '2th': 2, '4th': 3, '8th': 4, '16th': 5, '32th': 6}
    string_dict = {'N/A':0, 'G':1, 'D':2, 'A':3, 'E':4}
    with open(csv_file_path) as f:
        input_data = np.genfromtxt(f, delimiter=',', names=True, dtype=[('int'), ('int'), ('int'), ('<U10'), ('<U10'), ('<U10'), ('<U10')])
        pitches = input_data['pitch']
        starts = input_data['time_start']
        durations = input_data['duration']
        beat_types = np.array([beat_type_dict[bt] for bt in input_data['beat_type']], dtype=int)
        strings = np.array([string_dict[s] for s in input_data['string']], dtype=int)
        positions = np.array([0 if p == 'N/A' else int(p) for p in input_data['position']], dtype=int)
        fingers = np.array([0 if f == 'N/A' else int(f) for f in input_data['finger']], dtype=int)
    return {'pitches': pitches, 'starts': starts, 'durations': durations, 'beat_types': beat_types, 'strings': strings, 'positions': positions, 'fingers': fingers}


if __name__ == "__main__":
    # download pretrained model
    download_pretrained_model()

    # input data
    # pitches = [55, 57, 59, 60, 62, 64, 66, 67] # G scale
    # pitches = [62, 64, 66, 67, 69, 71, 73, 74] # D scale
    # pitches = [69, 71, 73, 74, 76, 78, 80, 81]  # A scale
    # pitches = [76, 78, 80, 81, 83, 85, 87, 88] # E scale
    # starts = [i * 256 for i in range(8)]
    # durations = [256 for _ in range(8)]
    # beat_types = [3 for _ in range(8)] # {'': 0, '1th': 1, '2th': 2, '4th': 3, '8th': 4, '16th': 5, '32th': 6}

    # load input from CSV
    input = read_csv(static.get_file("temp_notes.csv"))

    # inference, valid mode = {'basic', 'lowest', 'nearest'}
    model = violin_fingering_model()
    pred_str, pred_pos, pred_fin = model.inference(pitches=input['pitches'],
                                                   starts=input['starts'],
                                                   durations=input['durations'],
                                                   beat_types=input['beat_types'],
                                                   strings=input['strings'],
                                                   positions=input['positions'],
                                                   fingers=input['fingers'],
                                                   mode='basic')

    # print the results
    strings = ['N', 'G', 'D', 'A', 'E']
    n_notes = len(input['pitches'])
    print('pitch'.ljust(9), ''.join([pretty_midi.note_number_to_name(number).rjust(4) for number in input['pitches']]))
    print('string'.ljust(9), ''.join([strings[s].rjust(4) for s in pred_str[0, :n_notes]]))
    print('position'.ljust(9), ''.join([str(p).rjust(4) for p in pred_pos[0, :n_notes]]))
    print('finger'.ljust(9), ''.join([str(f).rjust(4) for f in pred_fin[0, :n_notes]]))