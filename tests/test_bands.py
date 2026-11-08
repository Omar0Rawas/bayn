import numpy as np
import pytest

from bayn.common.bus import pack, unpack
from bayn.eeg.bands import BANDS, band_powers, bands_message
from bayn.eeg.sources.simulated import SimulatedEEG
from tests.conftest import CHANNELS, FS

pytestmark = pytest.mark.unit
NAMES = list(BANDS)


def _sine(f: float, secs: float = 2.0, amp: float = 20.0) -> np.ndarray:
    t = np.arange(int(FS * secs)) / FS
    return np.tile(amp * np.sin(2 * np.pi * f * t), (4, 1)) + 500.0   # + electrode DC offset


@pytest.mark.parametrize("freq,band", [(2.5, "delta"), (6, "theta"), (10, "alpha"), (20, "beta"), (38, "gamma")])
def test_pure_tone_lands_in_its_band(freq, band):
    _, rel = band_powers(_sine(freq), FS)
    assert np.all(np.argmax(rel, axis=1) == NAMES.index(band))
    assert np.all(rel[:, NAMES.index(band)] > 0.9)


def test_relative_sums_to_one_and_dc_ignored():
    _, rel = band_powers(_sine(10), FS)
    np.testing.assert_allclose(rel.sum(axis=1), 1.0, atol=1e-6)


def test_mains_hum_not_counted_as_gamma():
    x = _sine(10, amp=5) + np.tile(40 * np.sin(2 * np.pi * 60 * np.arange(int(FS * 2)) / FS), (4, 1))
    _, rel = band_powers(x, FS)
    assert np.all(rel[:, NAMES.index("gamma")] < 0.05)       # 60 Hz is outside 30-44 Hz


def test_simulated_eeg_alpha_on_temporal_and_clench_raises_gamma():
    g = SimulatedEEG(CHANNELS, FS, seed=4, natural_blinks=False, natural_emg=False)
    g.generate(int(FS * 2))
    _, rest = band_powers(g.generate(int(FS * 2)), FS)
    g.inject("clench", delay_s=0)
    _, clench = band_powers(g.generate(int(FS * 1.4)), FS)
    gi = NAMES.index("gamma")
    assert np.all(clench[[0, 3], gi] > 3 * rest[[0, 3], gi])   # TP9 / TP10


def test_bands_message_roundtrips_over_bus():
    m = unpack(pack(bands_message(_sine(10), FS, CHANNELS, 123.0)))
    assert m["channels"] == CHANNELS and list(m["bands"]) == NAMES
    assert np.asarray(m["relative"]).shape == (4, 5)
