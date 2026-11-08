"""Muse 2 over native Bluetooth LE via BrainFlow (primary source).

No Windows pairing needed: BrainFlow (SimpleBLE) discovers and connects directly.
`board` can be any BrainFlow board name; SYNTHETIC_BOARD gives a hardware-free test.
"""

from __future__ import annotations

import logging

import numpy as np

from bayn.common.config import EEGConfig
from bayn.eeg.sources.base import ThreadedDeviceSource

log = logging.getLogger("bayn.eeg")


class BrainFlowSource(ThreadedDeviceSource):
    name = "muse"

    def __init__(self, eeg: EEGConfig) -> None:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        bf = eeg.brainflow
        try:
            self.board_id = BoardIds[bf.board].value
        except KeyError:
            raise ValueError(f"unknown BrainFlow board '{bf.board}'") from None
        fs = BoardShim.get_sampling_rate(self.board_id)
        super().__init__(eeg.channels, fs, stale_after_s=bf.stale_after_s)
        if bf.board != "MUSE_2_BOARD":
            self.name = bf.board.lower()

        self._eeg_rows = BoardShim.get_eeg_channels(self.board_id)[: len(eeg.channels)]
        self._ts_row = BoardShim.get_timestamp_channel(self.board_id)
        self._check_channel_names(eeg.channels)

        self._params = BrainFlowInputParams()
        self._params.serial_number = bf.serial_number
        self._params.mac_address = bf.mac_address
        self._params.timeout = bf.timeout_s
        self._target = bf.serial_number or bf.mac_address or "Muse 2"
        self._board = None
        if log.isEnabledFor(logging.DEBUG):
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

    def _check_channel_names(self, wanted: list[str]) -> None:
        from brainflow.board_shim import BoardShim

        try:
            names = BoardShim.get_eeg_names(self.board_id)[: len(wanted)]
        except Exception:  # some boards (synthetic) have no names
            return
        if self.board_id >= 0 and names and [n.upper() for n in names] != [w.upper() for w in wanted]:
            log.warning("board channel order %s differs from config %s", names, wanted)

    def _searching_message(self) -> str:
        return f"searching for {self._target}..."

    def _connect(self) -> None:
        from brainflow.board_shim import BoardShim

        self._board = BoardShim(self.board_id, self._params)
        self._board.prepare_session()          # BLE discovery + connect (blocks up to timeout_s)
        self._board.start_stream(45000)

    def _pull(self) -> tuple[np.ndarray, np.ndarray] | None:
        n = self._board.get_board_data_count()
        if n <= 0:
            return None
        d = self._board.get_board_data(n)
        return d[self._eeg_rows], d[self._ts_row]

    def _disconnect(self) -> None:
        if self._board is None:
            return
        try:
            if self._board.is_prepared():
                try:
                    self._board.stop_stream()
                finally:
                    self._board.release_session()
        finally:
            self._board = None
