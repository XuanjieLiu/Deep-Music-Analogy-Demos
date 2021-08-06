import pretty_midi as pm
import torch
import numpy as np

# SONG_PATH = "MIDI/jigs213.mid"  # 禁止访问
SONG_PATH = "MIDI/morris28.mid"  # 禁止访问
CHORD_PATH = "MIDI/chords/reelsa-c63.mid"
MELODY_PATH = "MIDI/melody/reelsa-c63.mid"
midi_pattern = pm.PrettyMIDI(SONG_PATH)
SONG_Error_PATH = "MIDI/reelsa-c29.mid"

aaa = [[[1,2],[2,9],[1,4]],[[3,4],[4,5],[3,7]]]
x1 = [[1,2],[2,9],[1,4],[3,4],[4,5],[3,7]]
x2 = [[1,2],[2,9],[1,4],[3,4]]
encode_x1 = torch.from_numpy(np.array(x1)).float()
encode_x2 = torch.from_numpy(np.array(x2)).float()
encode_x3 = torch.nn.ZeroPad2d(padding=(0, 0, 0, abs(encode_x1.size(0)-encode_x2.size(0))))(encode_x2)
encode_aaa = torch.from_numpy(np.array(aaa)).float()

def melody_to_numpy(fpath=MELODY_PATH, unit_time=0.125):
    music = pm.PrettyMIDI(fpath)
    notes = music.instruments[0].notes
    t = 0.
    roll = list()
#     print(notes[0], notes[-1])
    for note in notes:
#         print(t, note)
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

def chord_to_numpy(fpath=CHORD_PATH, unit_time=0.125):
    music = pm.PrettyMIDI(fpath)
    notes = music.instruments[0].notes
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

chroma = chord_to_numpy()
melody = melody_to_numpy()

instruments = midi_pattern.instruments
print(midi_pattern)
print(instruments)
print(chroma)