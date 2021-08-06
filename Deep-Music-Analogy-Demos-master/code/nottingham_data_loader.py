import random

import pretty_midi as pm
import torch
import os

PATH = "../../nottingham-dataset-master/nottingham-dataset-master/MIDI/"


# Reference: https://ldzhangyx.github.io/2020/01/13/music-practice/
def chord_to_numpy(fpath, unit_time=0.125):
    music = pm.PrettyMIDI(fpath)
    notes = music.instruments[1].notes
    max_end = 0.
    for note in notes:
        if note.end > max_end:
            max_end = note.end
    chroma = torch.zeros((int(round(max_end / unit_time)), 12))
    for note in notes:
        idx = int(round((note.start / unit_time)))
        n_unit = int(round((note.end - note.start) / unit_time))
        chroma[idx:idx + n_unit, note.pitch % 12] += 1
    return chroma


# Reference: https://ldzhangyx.github.io/2020/01/13/music-practice/
def melody_to_numpy(fpath, unit_time=0.125):
    music = pm.PrettyMIDI(fpath)
    notes = music.instruments[0].notes
    t = 0.
    roll = list()
    for note in notes:
        elapsed_time = note.start - t
        if elapsed_time > 0.:
            steps = torch.zeros((int(round(elapsed_time / unit_time)), 130))
            steps[range(int(round(elapsed_time / unit_time))), 129] += 1.
            roll.append(steps)
        n_units = int(round((note.end - note.start) / unit_time))
        steps = torch.zeros((n_units, 130))
        steps[0, note.pitch] += 1
        steps[range(1, n_units), 128] += 1
        roll.append(steps)
        t = note.end
    return torch.cat(roll, 0)


def get_a_specific_music_data(name):
    path = PATH + name
    melody = melody_to_numpy(path)
    chord = chord_to_numpy(path)
    if melody.size(0) > chord.size(0):
        chord = torch.nn.ZeroPad2d(padding=(0, 0, 0, abs(melody.size(0) - chord.size(0))))(chord)
    elif melody.size(0) < chord.size(0):
        melody = torch.nn.ZeroPad2d(padding=(0, 0, 0, abs(melody.size(0) - chord.size(0))))(melody)
    return melody.unsqueeze(0), chord.unsqueeze(0)


def get_a_N_step_data_from_a_specific_music(N, music_name):
    path = PATH + music_name
    print(path)
    melody = melody_to_numpy(path)
    chord = chord_to_numpy(path)
    min_len = min(melody.size(0), chord.size(0))
    batch_size = int(min_len / N)
    melody.resize_(batch_size, N, melody.size(-1))
    chord.resize_(batch_size, N, chord.size(-1))
    return melody, chord


class Dataloader():
    def __init__(self):
        self.f_list = os.listdir(PATH)#返回文件名

    def get_a_random_music_data(self):
        music_name = random.choice(self.f_list)
        print(music_name)
        return get_a_specific_music_data(music_name)

    def get_a_N_step_data_from_a_random_music(self, N):
        music_name = random.choice(self.f_list)
        print(music_name)
        return get_a_N_step_data_from_a_specific_music(N, music_name)




