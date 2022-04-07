import torch
import numpy as np
from mir_eval.separation import bss_eval_sources

class AudioMetircs():
    def __init__(self, reference, estimation, mix, sr):
        super(AudioMetircs, self).__init__()

        self.SISDR = sisdr(reference, estimation)
        self.SNR = snr(reference, estimation)
        self.SDRi = cal_SDRi(reference, estimation, mix)
        self.SISDRi = cal_SISNRi(reference, estimation, mix)
        self.SNRseg = SNRseg(reference, estimation, sr)

def sisdr(reference, estimation, sr=16000):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)
    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy

    projection = optimal_scaling * reference

    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return np.mean(10 * np.log10(ratio))

def snr(reference, estimation):
    numerator = np.sum(reference ** 2, axis=-1, keepdims=True)
    denominator = np.sum((estimation-reference) ** 2, axis=-1, keepdims=True)

    return np.mean(10 * np.log10(numerator / denominator))

def cal_SDR(reference, estimation, eps=1e-8):
    """
        Calculate Source-to-Distortion Ratio
        Args:
            reference:numpy.ndarray, [B, T]
            estimation:numpy.ndarray, [B, T]
    """
    origin_power = np.sum(reference**2, 1, keepdims=True) + eps # [B, 1]
    scale = np.sum(reference*estimation, 1, keepdims=True) / origin_power   # [B, 1]

    est_true = scale * reference    # [B, T]
    est_res = estimation - est_true

    true_power = np.sum(est_true**2, 1)
    res_power = np.sum(est_res**2, 1)

    return 10*np.log10(true_power) - 10*np.log10(res_power)

def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [N, T]
        src_est: numpy.ndarray, [N, T]
        mix: numpy.ndarray, [N, T]
        N 要求是不同的声源，这里将不同的声源换成不同的Batch_size
    Returns:
        average_SDRi
    """
    counter = mix.shape[0]
    # src_anchor = np.stack([mix, mix], axis=0)
    sdr = cal_SDR(src_ref, src_est)
    sdr0 = cal_SDR(src_ref, mix)
    avg_SDRi = np.sum(sdr - sdr0) / counter
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [N, T]
        src_est: numpy.ndarray, [N, T]
        mix: numpy.ndarray, [N, T]
    Returns:
        average_SISNRi
    """
    sisnr1 = sisdr(src_ref, src_est)
    sisnr1b = sisdr(src_ref, mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = sisnr1 - sisnr1b
    return avg_SISNRi

# Reference : https://github.com/schmiph2/pysepm

def SNRseg(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    eps = np.finfo(np.float64).eps

    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    MIN_SNR = -15  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extract_overlapped_windows(clean_speech, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extract_overlapped_windows(processed_speech, winlength, winlength - skiprate, hannWin)

    signal_energy = np.power(clean_speech_framed, 2).sum(-1)
    noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

    segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
    segmental_snr[segmental_snr < MIN_SNR] = MIN_SNR
    segmental_snr[segmental_snr > MAX_SNR] = MAX_SNR
    segmental_snr = segmental_snr[:-1]  # remove last frame -> not valid
    return np.mean(segmental_snr)

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    if window is not None:
        result = window * result
    return result

if __name__ == '__main__':
    a = np.random.randn(3, 4)
    b = np.random.randn(3, 4)
    mix = np.random.randn(3, 4)
    print(cal_SDRi(a, b, mix))
