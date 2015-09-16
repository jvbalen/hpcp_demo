
# coding: utf-8

# Harmonic Pitch Class Profile extraction

import numpy as np
from scipy.io import wavfile
from scipy.sparse import coo_matrix
from scipy.signal import spectrogram, convolve2d
import json
import sys


def main():
    """Compute Harmonic Pitch Class Profile (HPCP) features.
    Run from command line with filename (wav) as an argument
    or see HPCP.hpcp for other options."""
    print hpcp(sys.argv[1])


def hpcp(file_name,
         win_size=4096,
         hop_size=1024,
         window='blackman',
         precision='float32',
         f_min=100,
         f_max=5000,
         global_thr=80,  # in dB below the highest peak
         local_thr=30,  # in dB below 0
         bins_per_octave=12,
         whitening=True,
         filter_width=1/3.,  # in octaves
         harmonic_decay=0.6,
         harmonic_tolerance=2/3.,  # in semitones
         norm_frames=False,
         final_thr=0.,
         output='json'):
    """ Compute Harmonic Pitch Class Profile (HPCP) features.

    HPCP features are a type of chroma features, here implemented following
    Gomez' original proposal as close as possible [1], with some details
    borrowed from the summary in [2].

    HPCP computation involves seven main transformations (tuning frequency
    estimation not included):

    - spectrogram computation
    - discarding frequencies below `f_min` and above `f_max`
    - thresholding the spectrogram with a global and local (frame-wise) threshold
    - applying peak interpolation to obtain hi-res spectral peaks
    - computing a multi-octave pitch profile based on these peaks
    - summing together harmonics
    - folding to a single-octave HPCP

    Scipy is used to read audio, construct the sparse multidimensional pitch
    profile, and for efficient convolution.

    :param file_name (required)
    :param win_size: Short-Time Fourier Transform window size
    :param hop_size: Short-Time Fourier Transform hop size
    :param window: FFT window type (str)
    :param f_min
    :param f_max
    :param global_thr: global peak threshold (in dB below the highest peak)
    :param local_thr: frame-wise peak threshold (in dB below the highest peak)
    :param bins_per_octave
    :param whitening: whiten spectrum in the log-frequency domain for more
    timbre invariance (boolean)
    :param filter_width: width of whitening filter (in octaves)
    :param harmonic_decay: decay rate of weights of harmonics
    :param harmonic_tolerance: width of the cosine-weighted window around each
    of the harmonics
    :param norm_frames: normalize each HPCP frame as part of post-processing
    (boolean)
    :param final_thr: threshold and scale each HPCP frame as part of
    post-processing (between [0, 1])
    :param output: format of output ('json' or 'numpy')

    :return: hpcp features

    [1] Gomez, E. (2006). Tonal Description of Musical Audio Signals. PhD Thesis,
    Universitat Pompeu Fabra, Spain

    [2] Salamon, J., Gómez, E., & Bonada, J. (2011). Sinusoid extraction and
    salience function design for predominant melody estimation. In Proc. 14th
    Int. Conf. on Digital Audio Effects (DAFx-11), Paris, France (pp. 73–80).
    Retrieved from http://recherche.ircam.fr/pub/dafx11/Papers/14_e.pdf
    """

    # spectrogram
    y, sr = read_audio(file_name)
    Y, k, f, t = stft(y, sr, win_size=win_size, hop_size=hop_size, window=window, precision=precision)

    # prune spectrogram to [f_min, f_max]
    Y_lim, k, f = prune_spectrogram(Y, k, f, f_min=f_min, f_max=f_max)

    # threshold spectrogram based on dB magnitudes
    Y_dB = dB(Y_lim)
    Y_thr = global_thresholding(Y_dB, thr=global_thr)
    if local_thr < global_thr:
        Y_thr = local_thresholding(Y_thr, thr=local_thr)

    # peak interpolation
    Y_pks, F, peaks = spectral_peaks(Y_thr, k, sr, win_size)

    # multi-octave pitch profile based on linear magnitudes
    Y_lin = lin_mag(Y_pks, global_thr)
    pp = pitch_profile(Y_lin, F, peaks, bins_per_octave)
    if whitening:
        pp = whiten(pp, bins_per_octave=bins_per_octave, filter_width=filter_width)

    # harmonic summation
    hpp = sum_harmonics(pp, harmonic_decay=harmonic_decay,
                        harmonic_tolerance=harmonic_tolerance,
                        bins_per_octave=bins_per_octave)

    # fold to chromagram/hpcp
    pcp = fold_octaves(hpp, bins_per_octave=bins_per_octave)
    if norm_frames:
        pcp = normalize_frames(pcp, final_thr)

    return json.dumps({'chroma': pcp.tolist()}, indent=1) if output is 'json' else pcp


def read_audio(file_name):
    try:
        sr, y = wavfile.read(file_name)
    except IOError:
        print "File not found or inappropriate format. \n" \
              "Audio file should be in WAV format.\n"
        raise

    # if stereo, average channels
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # normalize
    y = y/np.max(y)
    return y, sr


def stft(x, sr, win_size=4096, hop_size=1024, window='blackman', precision='float32'):
    """ Short-Time Fourier Transform
    Wrapper on scipy.signal.spectrogram
    :param x: signal
    :param sr: sample rate
    :param win_size
    :param hop_size
    :param window: window type (str)
    :param precision: 'float32' or 'float64'
    :return:
    """
    f, t, X = spectrogram(x, sr, nperseg=win_size, noverlap=win_size-hop_size, window=window)
    X = X.astype(precision).T

    # keep bin numbers k
    k = np.arange(len(f))
    return X, k, f, t


def prune_spectrogram(X, k, f, f_min=100, f_max=5000):
    f_band = np.all([f > f_min, f < f_max], axis=0)
    return X[:, f_band], k[f_band], f[f_band]


def bin2hz(k, sr, win_size):
    return k*sr/win_size


def dB(x):
    return 20.0*np.log10(x)


def global_thresholding(X, thr=80):
    """ Set everything below max(X) - thr to zero.
    :param X: spectrogram
    :param thr: threshold (in dB)
    :return: thresholded spectrogram
    """
    X = X - np.max(X) + thr
    X[X < 0] = 0
    return X


def local_thresholding(X, thr=30):
    """ For every frame, set everything below max(frame) - thr to zero.
    :param X: spectrogram
    :param thr: threshold (in dB)
    :return: thresholded spectrogram
    """
    n_frames, n_bins = X.shape
    X[X < np.tile(np.max(X, axis=1).reshape((-1, 1)) - thr, (1, n_bins))] = 0
    return X


def spectral_peaks(X, k, sr, win_size):
    """ Compute frequency-corrected spectral peaks.

    Compute frequency- and amplitude-corrected spectral peaks using peak
    interpolation. As proposed by Gomez and conveniently summarized in [1].

    [1] Salamon, J., Gómez, E., & Bonada, J. (2011). Sinusoid extraction and
    salience function design for predominant melody estimation. In Proc. 14th
    Int. Conf. on Digital Audio Effects (DAFx-11), Paris, France (pp. 73–80).
    Retrieved from http://recherche.ircam.fr/pub/dafx11/Papers/14_e.pdf

    :param X: spectrogram
    :param k: bin numbers
    :param sr: sample rate
    :param win_size: Short-Time Fourier Transform window size
    :return A: corrected peak amplitudes
    :return F: corrected peak frequencies
    :return peaks: spectrogram peaks
    """

    n_frames, n_bins = X.shape
    precision = X.dtype

    A1 = np.zeros((n_frames, n_bins), dtype=precision)
    A2 = np.zeros((n_frames, n_bins), dtype=precision)
    A3 = np.zeros((n_frames, n_bins), dtype=precision)
    A1[:,1:-1] = X[:,:-2]
    A2[:,1:-1] = X[:,1:-1]
    A3[:,1:-1] = X[:,2:]
    peaks = np.all([A2>A1, A2>A3], axis=0)

    # Bin number of each peak
    K = k * peaks

    # Compute deviations D of spectral peaks, in bins
    D = np.zeros((n_frames, n_bins), dtype=precision)
    D[peaks] = .5 * (A1[peaks] - A3[peaks]) / (A1[peaks] - 2*A2[peaks] + A3[peaks])

    # Vompute adjusted frequencies and amplitudes
    F = bin2hz(K + D, sr, win_size)
    A = np.zeros((n_frames, n_bins), dtype=precision)
    A[peaks] = A2[peaks] - D[peaks]/4*(A1[peaks]-A3[peaks])

    return A, F, peaks


def lin_mag(x, x_max):
    """ Linear amplitude (magnitude) from dB amplitude (inverse of dB())
    :param x: amplitude in dB
    :param x_max: maximum amplitude
    :return: linear amplitude
    """
    return 10**((x - x_max)/20)


def pitch_profile(X, F, peaks, bins_per_octave):
    """ Construct multi-octave pitch profile
    :param X: spectral peak amplitudes (corrected)
    :param F: spectral peak frequencies (corrected)
    :param bins_per_octave: pitch profile resolution
    :return: multi-octave pitch profile
    """
    n_frames, n_bins = X.shape
    T = np.ones((n_frames, n_bins)) * np.arange(n_frames).reshape((-1, 1))  # t in frames, not seconds
    pitch = hz2midi(F)
    pitch_in_bins = bins_per_octave * pitch / 12

    # fill sparse matrix with spectral peak amplitudes in the right bins
    pp = coo_matrix((X[peaks], (T[peaks], pitch_in_bins[peaks])))
    return pp.toarray()


def hz2midi(f):
    m = np.zeros(f.shape)
    m[f > 0] = 69 + 12.*np.log2(f[f > 0]/440)
    return m


def whiten(X, bins_per_octave, filter_width=1/3.):
    """ Pitch profile whitening (spectral whitening in the log-frequency domain)
    :param X: pitch profile or other constant-Q profile
    :param bins_per_octave: pitch profile resolution
    :param filter_width: width of the whitening filter
    :return: whitened pitch profile
    """
    filter_width_in_bins = int(bins_per_octave * filter_width)

    # moving average filter kernel
    filter_kernel = np.ones((1, filter_width_in_bins), dtype=X.dtype)
    filter_kernel = filter_kernel / np.sum(filter_kernel)

    # subtract moving average
    X = X - convolve2d(X, filter_kernel, mode='same')
    X[X < 0] = 0
    return X


def sum_harmonics(X, harmonic_decay=.6, harmonic_tolerance=1, bins_per_octave=120):
    w = harmonic_summation_kernel(harmonic_decay=harmonic_decay,
                                  harmonic_tolerance=harmonic_tolerance,
                                  bins_per_octave=bins_per_octave)
    w = w.astype(X.dtype).reshape((1,-1))

    # sum harmonics in X using convolution with precomputed kernel w
    return convolve2d(X, w, mode='same')


def harmonic_summation_kernel(harmonic_decay=.6, harmonic_tolerance=1,
                              bins_per_octave=120, n_octaves=4):
    """ Compute harmonic summation kernel using the parameters proposed by Gomez.
    Harmonics are weighted according to their harmonic number n and the harmonic
    deviation d.
    w(n) is given by a geometric series, w(d) is given by a cos^2 window.
    w(d,n) = w(n) * w(d)

    :param harmonic_decay: model decay rate of successive harmonics
    :param harmonic_tolerance: maximum allowed harmonic deviation
    :param bins_per_octave: pitch profile resolution
    :param n_octaves: size of the kernel
    :return:
    """

    # f/f0 (log, in octaves) for a linspace of constant Q bins symmetrically around f0
    f_ratio_octaves = 1. * np.arange(-n_octaves*bins_per_octave, n_octaves*bins_per_octave+1) / bins_per_octave

    # f/f0 (in Hz)
    f_ratio = 2**f_ratio_octaves

    # harmonic number and harmonic deviation
    n_harm = np.round(f_ratio)
    d_harm = abs(f_ratio - n_harm)

    w = cosine_window(d_harm, tol=harmonic_tolerance) * attenuation(n_harm, r=harmonic_decay)
    return w / np.sum(w)


def attenuation(n, r=.6):
    n = np.array(np.round(n))
    w = np.zeros(n.shape)
    w[n>0] = r**(n[n>0]-1)
    return w


def cosine_window(d, tol=1.):
    # width of the cosine-weighted window around each of the harmonics
    width = np.log(2**(tol/12.))
    w = np.zeros(d.shape)
    w[d < width] = np.cos(d[d < width]*(np.pi/2)/width)**2
    return w


def fold_octaves(X, bins_per_octave):
    n_frames, n_bins = X.shape

    # fold multi-octave pitch profile at every C
    folds = np.arange(0, n_bins, bins_per_octave)  # every C
    return np.array([X[:,fold:fold+bins_per_octave] for fold in folds[:-1]]).sum(axis=0)


def normalize_frames(X, thr):
    X = X - np.min(X, axis=1).reshape((-1,1))
    X_max = np.max(X, axis=1)
    X = X[X_max >0] / (X_max[X_max > 0]).reshape((-1,1))
    if thr > 0:
        X = (1-thr) * (X-thr) * (X>thr)
    return X


if __name__ == '__main__':
    main()