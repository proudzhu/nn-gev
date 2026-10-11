"""NN-GEV beamforming for raw 5-channel PCM dumps (16 kHz, 16 bit, interleaved).

Uses the first two channels of each PCM file to run the NN-GEV beamformer
(BLSTM mask estimation + GEV). Writes a 6-channel WAV per file containing
the original 5 channels followed by the beamformed channel.
"""
import argparse
import os
import time

import numpy as np
import chainer
from chainer import Variable
from chainer import serializers
from scipy.io import wavfile
from tqdm import tqdm

from fgnt.beamforming import gev_wrapper_on_masks
from fgnt.signal_processing import stft, istft
from fgnt.utils import mkdir_p
from nn_models import BLSTMMaskEstimator

N_CHANNELS = 5
FS = 16000


def load_pcm(pcm_path):
    raw = np.fromfile(pcm_path, dtype=np.int16)
    assert raw.size % N_CHANNELS == 0, (
        'Sample count {} is not a multiple of {} channels'.format(
            raw.size, N_CHANNELS))
    # (N_CHANNELS, samples)
    return raw.reshape(-1, N_CHANNELS).T


def beamform(audio, model):
    """audio: (5, samples) int16. Returns beamformed signal (samples,)."""
    bf_input = audio[:2].astype(np.float32) / 32768.

    # (frames, channels, bins)
    Y = stft(bf_input, time_dim=1).transpose((1, 0, 2))
    Y_var = Variable(np.abs(Y).astype(np.float32))
    N_masks, X_masks = model.calc_masks(Y_var)
    N_mask = np.median(N_masks.data, axis=1)
    X_mask = np.median(X_masks.data, axis=1)

    Y_hat = gev_wrapper_on_masks(Y, N_mask, X_mask)
    return istft(Y_hat)[:audio.shape[1]]


def to_int16(signal):
    scaled = signal * 32768.
    peak = np.max(np.abs(scaled)) if scaled.size else 0.
    if peak > 32767.:
        print('    beamformed channel normalized (peak was {:.1f})'.format(
            peak))
        scaled *= 32767. / peak
    return np.clip(np.round(scaled), -32768, 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(
        description='NN-GEV beamforming for 5ch PCM dumps')
    parser.add_argument('input', help='PCM file or directory with PCM files')
    parser.add_argument('model', help='Trained BLSTM model file')
    parser.add_argument('output_dir', help='Directory for the output WAVs')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        pcm_files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input) if f.lower().endswith('.pcm'))
    else:
        pcm_files = [args.input]
    mkdir_p(args.output_dir)

    model = BLSTMMaskEstimator()
    serializers.load_hdf5(args.model, model)
    chainer.no_backprop_mode()

    for pcm_path in tqdm(pcm_files):
        print('Processing {}'.format(pcm_path))
        t_start = time.time()

        audio = load_pcm(pcm_path)
        y_bf = beamform(audio, model)

        # 6 channels: original 5 ch + beamformed ch
        out = np.concatenate(
            [audio.T, to_int16(y_bf)[:, None]], axis=1).astype(np.int16)
        out_path = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(pcm_path))[0] + '_gev.wav')
        wavfile.write(out_path, FS, out)
        print('  wrote {} ({} s audio, {:.1f} s processing)'.format(
            out_path, audio.shape[1] / FS, time.time() - t_start))


if __name__ == '__main__':
    main()
