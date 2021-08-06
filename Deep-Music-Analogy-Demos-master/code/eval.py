import json
import torch
import os
from model import VAE
from nottingham_data_loader import Dataloader, get_a_N_step_data_from_a_specific_music
from torch_to_midi import *

PITCH_TARGET = "ashover1.mid"
RHYTHM_TARGET = "ashover2.mid"
RECON_PATH = "samples/"


def recon_path():
    return f'{RECON_PATH}/PITCH_{PITCH_TARGET.split(".")[0]} - RHYTHM_{RHYTHM_TARGET.split(".")[0]}.mid'


with open('model_config.json') as f:
    args = json.load(f)
MODEL_PATH = 'params/{}.pt'.format(args['name'])


class Eval:
    def __init__(self):
        self.music_time = args['time_step']
        self.model = self.init_model()
        self.dl = Dataloader()

    def init_model(self):
        model = VAE(130, args['hidden_dim'], 3, 12, args['pitch_dim'],
                    args['rhythm_dim'], self.music_time)
        if args['if_parallel']:
            # model = torch.nn.DataParallel(model, device_ids=[0, 1])
            model = torch.nn.DataParallel(model, device_ids=[0])
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH))
            print(f"Model is loaded.")
        else:
            print("No model found, please train.")
        if torch.cuda.is_available():
            print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
            model.cuda()
        else:
            print('CPU mode')
        model.eval()
        return model.module

    def get_recon_rhythm(self, melody, chord):
        if torch.cuda.is_available():
            chord = chord.cuda()
            melody = melody.cuda()
        dis1, dis2 = self.model.encoder(melody, chord)
        z2 = dis2.rsample()
        recon_rhythm = self.model.rhythm_decoder(z2)
        return recon_rhythm

    def get_final_recon(self, melody, chord, recon_rhythm):
        if torch.cuda.is_available():
            chord = chord.cuda()
            melody = melody.cuda()
        dis1, dis2 = self.model.encoder(melody, chord)
        z1 = dis1.rsample()
        recon = self.model.final_decoder(z1, recon_rhythm, chord)
        return recon

    def pitch_rhythm_fusion(self, pitch_target_name, rhythm_target_name):
        melody_p, chord_p = get_a_N_step_data_from_a_specific_music(self.music_time, pitch_target_name)
        melody_r, chord_r = get_a_N_step_data_from_a_specific_music(self.music_time, rhythm_target_name)
        min_batch = min(melody_p.size(0), melody_r.size(0))
        melody_p = melody_p[:min_batch, :, :]
        chord_p = chord_p[:min_batch, :, :]
        melody_r = melody_r[:min_batch, :, :]
        chord_r = chord_r[:min_batch, :, :]
        recon_rhythm = self.get_recon_rhythm(melody_r, chord_r)
        final_recon = self.get_final_recon(melody_p, chord_p, recon_rhythm)
        return final_recon


if __name__ == "__main__":
    if not os.path.isdir('samples'):
        os.mkdir('samples')
    my_eval = Eval()
    final_recon = my_eval.pitch_rhythm_fusion(PITCH_TARGET, RHYTHM_TARGET)
    batch_roll_to_midi(final_recon, recon_path())
    print("yeah")
