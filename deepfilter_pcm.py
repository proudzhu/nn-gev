"""DeepFilterNet3 enhancement + DFN-mask GEV beamforming for raw
5-channel PCM dumps (16 kHz, 16 bit, interleaved).

Runs DeepFilterNet3 on the first channel (spectral mask + deep filtering)
and uses the DFN-estimated effective mask to drive GEV, MVDR, and PnP MVDR
beamformers (CH1 + CH2). PnP MVDR follows Chang et al. (ICASSP 2024): the
MVDR objective is regularized by RED (Regularization by Denoising) and
solved by ADMM-style variable splitting with DFN plugged in as the speech
prior denoiser. Each beamformer output is additionally post-enhanced by a
single-channel DFN pass. Writes a 12-channel WAV per file: original 5
channels, DFN enhanced CH1, mask GEV, GEV+DFN, mask MVDR, MVDR+DFN,
PnP MVDR, PnP MVDR+DFN.
"""
import argparse
import os
import time

import numpy as np
import torch
from df.enhance import init_df
from df.io import resample
from df.model import ModelParams
from df.utils import as_complex, as_real, get_device, get_norm_alpha
from libdf import erb, erb_norm, unit_norm
from scipy.io import wavfile
from tqdm import tqdm

from fgnt.beamforming import (
    apply_beamforming_vector,
    condition_covariance,
    get_mvdr_vector,
    get_pca_vector,
    get_power_spectral_density_matrix,
    gev_wrapper_on_masks,
    phase_correction,
)

N_CHANNELS = 5
FS = 16000
EPS = 1e-8
TARGET_RMS = 0.05  # -26 dBFS, roughly the level DFN was trained on


def mvdr_wrapper_on_masks(mix, noise_mask, target_mask):
    """Mask-based MVDR. mix: [T, C, F], masks: [T, F]. Returns [T, F].

    The ATF/RTF vector is the principal eigenvector of the target PSD
    matrix (max-SNR style steering vector estimate).
    """
    org_dtype = mix.dtype
    mix = mix.astype(np.complex128)
    mix = mix.T  # [F, C, T]
    noise_mask = noise_mask.T
    target_mask = target_mask.T

    target_psd_matrix = get_power_spectral_density_matrix(
        mix, target_mask, normalize=True)
    noise_psd_matrix = get_power_spectral_density_matrix(
        mix, noise_mask, normalize=True)
    noise_psd_matrix = condition_covariance(noise_psd_matrix, 1e-6)

    atf_vector = get_pca_vector(target_psd_matrix)  # [F, C]
    w_mvdr = get_mvdr_vector(atf_vector, noise_psd_matrix)  # [F, C]
    w_mvdr = phase_correction(w_mvdr)

    output = apply_beamforming_vector(w_mvdr, mix)  # [F, T]
    return output.astype(org_dtype).T


def pnp_mvdr(mix, target_mask, omega, rho=50., mu=0.4, iters=3):
    """PnP MVDR beamforming following Chang et al., "Plug-and-play MVDR
    beamforming for speech separation", ICASSP 2024 (Algorithm 1).

    The MVDR problem is regularized with RED (Regularization by
    Denoising); ADMM-style variable splitting alternates between an
    analytical beamformer update (w), a denoiser-driven speech estimate
    (R), and auxiliary variables (V, P):

        w  = argmin ||w^H X||^2 + rho*||w^H X - (R - P + V)||^2
             s.t. w^H a = 1                              (per bin)
        R  = mu*R_hat + (1 - mu)*Omega(R_hat),  R_hat = Shat + P - V
        V  = Shat - R + P
        P  = P + Shat - R - V

    mix: [T, C, F] complex STFT of the observations (df_sr grid)
    target_mask: [T, F] speech mask, used to estimate the steering
        vector a (principal eigenvector of the target PSD matrix)
    omega: callable mapping a [F, T] complex spectrogram to its
        DNN-denoised counterpart [F, T] (the plugged-in prior)
    rho: penalty parameter of the variable splitting method
    mu: fixed-point trade-off parameter (rho / (rho + alpha))
    Returns the final speech estimate R as [T, F].
    """
    X = mix.astype(np.complex128).T  # [F, C, T]
    F, C, T = X.shape
    tm = target_mask.T  # [F, T]

    # Steering vector: principal eigenvector of the target PSD matrix
    tgt_psd = get_power_spectral_density_matrix(X, tm, normalize=True)
    a = get_pca_vector(tgt_psd)  # [F, C]

    # Frame-averaged observation covariance per bin and its inverse.
    # (Scaling Rx and d by 1/T leaves w unchanged, so rho keeps the
    # paper's raw-sum meaning.)
    Rx = np.einsum('fct,fet->fce', X, X.conj()) / T
    Rx = condition_covariance(Rx, 1e-6)
    Rx_inv = np.linalg.inv(Rx)
    u = np.einsum('fcd,fd->fc', Rx_inv, a)            # Rx^-1 a
    h = np.einsum('fc,fc->f', a.conj(), u).real       # a^H Rx^-1 a

    R = np.zeros((F, T), np.complex128)
    V = np.zeros_like(R)
    P = np.zeros_like(R)
    for _ in range(iters):
        # --- Step 1: analytical beamformer update (per frequency bin) ---
        c_lin = R - P + V                             # match target
        d = np.einsum('fct,ft->fc', X, c_lin.conj()) / T
        q = np.einsum('fc,fc->f', a.conj(),
                      np.einsum('fcd,fd->fc', Rx_inv, d))
        lam = (rho * q - (1. + rho)) / h
        w = np.einsum('fcd,fd->fc', Rx_inv,
                      rho * d - lam[:, None] * a) / (1. + rho)
        Shat = np.einsum('fc,fct->ft', w.conj(), X)   # beamformed output

        # --- Step 2: RED fixed point with the plugged-in denoiser ---
        R_hat = Shat + P - V
        R = mu * R_hat + (1. - mu) * omega(R_hat)

        # --- Steps 3 + 4: auxiliary variable updates ---
        V = Shat - R + P
        P = P + Shat - R - V
    return R.T.astype(mix.dtype)


def load_pcm(pcm_path):
    raw = np.fromfile(pcm_path, dtype=np.int16)
    assert raw.size % N_CHANNELS == 0, (
        'Sample count {} is not a multiple of {} channels'.format(
            raw.size, N_CHANNELS))
    # (N_CHANNELS, samples)
    return raw.reshape(-1, N_CHANNELS).T


def to_int16(signal):
    scaled = signal * 32768.
    peak = np.max(np.abs(scaled)) if scaled.size else 0.
    if peak > 32767.:
        print('    channel normalized (peak was {:.1f})'.format(peak))
        scaled *= 32767. / peak
    return np.clip(np.round(scaled), -32768, 32767).astype(np.int16)


def df_features(audio_np, df_state, nb_df, device=None):
    """STFT + ERB/spectral features, mirrors df.enhance.df_features."""
    spec = df_state.analysis(audio_np)  # [C, T, F]
    a = get_norm_alpha(False)
    erb_fb = df_state.erb_widths()
    erb_feat = torch.as_tensor(erb_norm(erb(spec, erb_fb), a)).unsqueeze(1)
    spec_feat = as_real(
        torch.as_tensor(unit_norm(spec[..., :nb_df], a)).unsqueeze(1))
    spec_r = as_real(torch.as_tensor(spec).unsqueeze(1))
    if device is not None:
        spec_r = spec_r.to(device)
        erb_feat = erb_feat.to(device)
        spec_feat = spec_feat.to(device)
    return spec, spec_r, erb_feat, spec_feat


@torch.no_grad()
def enhance_and_mask(model, df_state, audio):
    """Run DFN on audio [C, T] (model sr).

    Returns:
        enhanced: [C, T] enhanced time domain audio
        eff_mask: [T, F] effective speech mask (median over channels)
        spec: [C, T, F] complex STFT of the (padded) input
    """
    model.eval()
    if hasattr(model, 'reset_h0'):
        model.reset_h0(batch_size=audio.shape[0], device=get_device())
    n_fft, hop = df_state.fft_size(), df_state.hop_size()
    orig_len = audio.shape[-1]
    audio_pad = torch.nn.functional.pad(audio, (0, n_fft))

    nb_df = getattr(model, 'nb_df', getattr(model, 'df_bins',
                                            ModelParams().nb_df))
    spec, spec_r, erb_feat, spec_feat = df_features(
        audio_pad.numpy(), df_state, nb_df, device=get_device())
    spec_e, _, _, _ = model(spec_r.clone(), erb_feat, spec_feat)
    spec_e = as_complex(spec_e.squeeze(1)).cpu()  # [C, T, F]

    # Full-band effective speech gain of the DFN enhancement
    spec_t = torch.as_tensor(spec)
    eff_mask = (spec_e.abs() / spec_t.abs().add(EPS)).clamp(0, 1)
    eff_mask = eff_mask.median(dim=0).values.numpy()  # [T, F]

    enhanced = torch.as_tensor(
        df_state.synthesis(spec_e.numpy()))
    d = n_fft - hop
    enhanced = enhanced[:, d:orig_len + d]
    return enhanced, eff_mask, spec


def main():
    parser = argparse.ArgumentParser(
        description='DFN3 + DFN-mask GEV for 5ch PCM dumps')
    parser.add_argument('input', help='PCM file or directory with PCM files')
    parser.add_argument('output_dir', help='Directory for the output WAVs')
    parser.add_argument('--atten-lim', type=float, default=None,
                        help='Noise attenuation limit in dB (DFN channel)')
    parser.add_argument('--post-filter', action='store_true',
                        help='Enable DFN post filter')
    parser.add_argument('--pnp-iters', type=int, default=3,
                        help='PnP MVDR outer iterations (default: 3)')
    parser.add_argument('--pnp-rho', type=float, default=50.,
                        help='PnP MVDR penalty parameter rho '
                             '(paper: 5-95, default: 50)')
    parser.add_argument('--pnp-mu', type=float, default=0.4,
                        help='PnP MVDR fixed-point trade-off mu '
                             '(paper: 0.25-0.5, default: 0.4)')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        pcm_files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input) if f.lower().endswith('.pcm'))
    else:
        pcm_files = [args.input]
    os.makedirs(args.output_dir, exist_ok=True)

    model, df_state, _ = init_df(post_filter=args.post_filter,
                                 log_level='ERROR')
    df_sr = df_state.sr()
    n_fft, hop = df_state.fft_size(), df_state.hop_size()

    for pcm_path in tqdm(pcm_files):
        print('Processing {}'.format(pcm_path))
        t_start = time.time()

        audio = load_pcm(pcm_path)
        n_samples = audio.shape[1]

        # DFN is level-sensitive: normalize input RMS to TARGET_RMS and
        # undo the gain afterwards. The same gain is applied to both
        # channels to preserve the inter-channel level ratio for GEV.
        rms = float(np.sqrt(np.mean(
            (audio[:2].astype(np.float32) / 32768.) ** 2)))
        gain = np.clip(TARGET_RMS / max(rms, 1e-8), 1e-3, 1e3)

        # --- DFN enhancement on CH1 + mask estimation on CH1/CH2 ---
        x = torch.from_numpy(audio[:2].astype(np.float32) / 32768. * gain)
        x = resample(x, FS, df_sr)
        enhanced, eff_mask, spec = enhance_and_mask(model, df_state, x)

        # --- Beamforming driven by the DFN mask (CH1 + CH2) ---
        mix = spec.transpose(1, 0, 2)  # [T, C, F] complex
        n48 = x.shape[-1]
        d = n_fft - hop

        def synth_trim(y_hat):
            """[T, F] spec -> time domain at df_sr, delay-compensated."""
            y = df_state.synthesis(
                np.ascontiguousarray(y_hat[None].astype(np.complex64)))[0]
            return y[d:d + n48]

        def dfn_post(y):
            """Single-channel DFN pass at df_sr, level-normalized.

            Returns (enhanced time signal, effective [T, F] speech mask).
            The mask is scale invariant, the signal is restored to input
            scale.
            """
            r = float(np.sqrt(np.mean(y ** 2)))
            g = np.clip(TARGET_RMS / max(r, 1e-8), 1e-3, 1e3)
            e, m, _ = enhance_and_mask(
                model, df_state,
                torch.from_numpy(
                    np.ascontiguousarray(y * g, dtype=np.float32))[None])
            return e[0].numpy() / g, m

        def bf(bf_func, m):
            return bf_func(mix, noise_mask=1. - m, target_mask=m)

        y_gev = synth_trim(bf(gev_wrapper_on_masks, eff_mask))
        y_mvdr = synth_trim(bf(mvdr_wrapper_on_masks, eff_mask))

        # --- PnP MVDR (Chang et al., ICASSP 2024): DFN plugged in as the
        # RED denoiser. Omega: [F, T] spec -> DFN-denoised [F, T] spec
        # via a time-domain round trip (level-normalized DFN pass).
        def omega(spec_FT):
            y = synth_trim(spec_FT.T)
            y_den, _ = dfn_post(y)
            spec_den = df_state.analysis(np.ascontiguousarray(
                np.pad(y_den, (0, n_fft))[None].astype(np.float32)))[0]
            return spec_den.T

        R_pnp = pnp_mvdr(mix, eff_mask, omega,
                         rho=args.pnp_rho, mu=args.pnp_mu,
                         iters=args.pnp_iters)
        y_pnp = synth_trim(R_pnp)

        # DFN post-enhancement of the beamformer outputs
        y_gev_dfn, _ = dfn_post(y_gev)
        y_mvdr_dfn, _ = dfn_post(y_mvdr)
        y_pnp_dfn, _ = dfn_post(y_pnp)

        # All enhanced channels back to 16 kHz
        def to_16k(y):
            return resample(
                torch.from_numpy(
                    np.ascontiguousarray(y, dtype=np.float32)[None]),
                df_sr, FS).numpy()[0]

        y_dfn = to_16k(enhanced[0].numpy())
        y_gev, y_gev_dfn = to_16k(y_gev), to_16k(y_gev_dfn)
        y_mvdr, y_mvdr_dfn = to_16k(y_mvdr), to_16k(y_mvdr_dfn)
        y_pnp, y_pnp_dfn = to_16k(y_pnp), to_16k(y_pnp_dfn)

        # Optional attenuation limit on the DFN channel
        if args.atten_lim is not None and abs(args.atten_lim) > 0:
            lim = 10 ** (-abs(args.atten_lim) / 20)
            y_dfn = y_dfn * (1 - lim) + (audio[0] / 32768.) * lim

        # Undo the input gain
        y_dfn /= gain
        y_gev /= gain
        y_mvdr /= gain
        y_pnp /= gain

        # Align lengths
        def align(y):
            if y.shape[0] >= n_samples:
                return y[:n_samples]
            return np.pad(y, (0, n_samples - y.shape[0]))

        y_dfn = align(y_dfn)
        y_gev, y_gev_dfn = align(y_gev), align(y_gev_dfn)
        y_mvdr, y_mvdr_dfn = align(y_mvdr), align(y_mvdr_dfn)
        y_pnp, y_pnp_dfn = align(y_pnp), align(y_pnp_dfn)

        # 12 channels: original 5 ch + DFN CH1 + mask GEV + GEV+DFN
        #               + mask MVDR + MVDR+DFN + PnP MVDR + PnP MVDR+DFN
        out = np.concatenate(
            [audio.T,
             to_int16(y_dfn)[:, None],
             to_int16(y_gev)[:, None],
             to_int16(y_gev_dfn)[:, None],
             to_int16(y_mvdr)[:, None],
             to_int16(y_mvdr_dfn)[:, None],
             to_int16(y_pnp)[:, None],
             to_int16(y_pnp_dfn)[:, None]], axis=1).astype(np.int16)
        out_path = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(pcm_path))[0] + '_dfn_gev.wav')
        try:
            wavfile.write(out_path, FS, out)
        except PermissionError:
            print('  WARNING: {} is locked (open in another program?), '
                  'skipped'.format(out_path))
            continue
        print('  wrote {} ({} s audio, {:.1f} s processing)'.format(
            out_path, n_samples / FS, time.time() - t_start))


if __name__ == '__main__':
    main()
