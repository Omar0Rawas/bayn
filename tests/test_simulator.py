import numpy as np
import pytest

from bayn.eeg.filters import StreamingFilter
from bayn.eeg.quality import robust_amplitude
from bayn.eeg.sources.simulated import SimulatedEEG
from tests.conftest import CHANNELS, FS, band_rms

pytestmark = pytest.mark.unit


def test_shape_dtype_and_plausible_amplitude(gen):
    x = gen.generate(int(FS * 10))
    assert x.shape == (4, int(FS * 10)) and x.dtype == np.float32
    y = StreamingFilter(FS, 4).process(x)[:, int(FS):]   # skip filter warm-up
    amp = robust_amplitude(y)
    assert np.all((amp > 3) & (amp < 40)), amp            # typical scalp EEG range


def test_chunked_equals_continuous():
    # distractors consume random numbers at chunk-dependent moments, so disable them here
    a = SimulatedEEG(CHANNELS, FS, seed=7, natural_blinks=False, natural_emg=False)
    b = SimulatedEEG(CHANNELS, FS, seed=7, natural_blinks=False, natural_emg=False)
    whole = a.generate(1024)
    parts = np.hstack([b.generate(32) for _ in range(32)])
    np.testing.assert_allclose(whole, parts, atol=1e-3)


def test_alpha_peak_present(gen):
    from scipy.signal import welch

    f, p = welch(gen.generate(int(FS * 20)), fs=FS, nperseg=int(FS * 2), axis=-1)
    band = (f > 6) & (f < 14)
    peak_f = f[band][np.argmax(p[:, band].mean(axis=0))]
    assert 9 <= peak_f <= 11


def test_clench_raises_emg_on_temporal_channels(gen):
    gen.generate(int(FS * 2))
    base = band_rms(gen.generate(int(FS * 2)), FS, 55, 100)
    gen.inject("clench", delay_s=0)
    burst = band_rms(gen.generate(int(FS * 1.4)), FS, 55, 100)
    tp = [0, 3]
    assert np.all(burst[tp] > 5 * base[tp])
    assert np.all(burst[tp] > burst[[1, 2]])              # temporal > frontal


def test_double_blink_gives_two_frontal_peaks(gen):
    from scipy.signal import find_peaks

    gen.generate(int(FS * 2))
    gen.inject("double_blink", delay_s=0.2)
    x = gen.generate(int(FS * 2))
    y = StreamingFilter(FS, 4, band=(0.5, 10), notch="off").process(x)
    for ch in (1, 2):                                      # AF7, AF8
        peaks, _ = find_peaks(y[ch], height=80, distance=int(0.2 * FS))
        assert len(peaks) == 2, peaks
        assert 0.3 <= (peaks[1] - peaks[0]) / FS <= 0.55


def test_unknown_gesture_rejected(gen):
    with pytest.raises(ValueError):
        gen.inject("wink")

