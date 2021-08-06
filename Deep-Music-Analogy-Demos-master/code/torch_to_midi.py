import pretty_midi as pm
import numpy as np


# https://ldzhangyx.github.io/2020/01/13/music-practice/
def sample_roll_to_piano(sample_roll, piano):
    if len(piano.notes) == 0:
        t = 0
    else:
        t = piano.notes[-1].end
    for i in sample_roll:
        if 'torch' in str(type(i)):
            pitch = int(i.max(0)[1])
        else:
            pitch = int(np.argmax(i))
        if pitch < 128:
            note = pm.Note(
                velocity=100, pitch=pitch, start=t, end=t + 1 / 8)
            t += 1 / 8
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pm.Note(
                    velocity=100, pitch=int(p), start=0, end=t)
            note = pm.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + 1 / 8)
            piano.notes.append(note)
            t += 1 / 8
        elif pitch == 129:
            t += 1 / 8


def batch_roll_to_piano(batch_roll, piano):
    for i in range(0, batch_roll.size(0)):
        sample_roll_to_piano(batch_roll[i], piano)


def batch_roll_to_midi(batch_roll, output='sample.mid'):
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)
    batch_roll_to_piano(batch_roll, piano)
    music.instruments.append(piano)
    music.write(output)
