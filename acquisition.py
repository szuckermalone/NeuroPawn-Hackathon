import time
import numpy as np


class Source:
    """Read each sample once. The first column is its host timestamp."""
    def __init__(self, mode, port, imu, seed, demo_fatigue=False):
        self.mode = mode
        self.board = None
        self.rng = np.random.default_rng(seed)
        self.demo_fatigue = demo_fatigue
        self.start_time = time.time()
        if mode == "synthetic":
            self.fs = 125
            self.last = time.time()
            return
        from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
        name = "NEUROPAWN_KNIGHT_BOARD" + ("_IMU" if imu else "")
        if not hasattr(BoardIds, name):
            raise RuntimeError(f"BrainFlow installation lacks {name}; check NeuroPawn SDK.")
        board_id = getattr(BoardIds, name).value
        params = BrainFlowInputParams()
        params.serial_port = port
        params.other_info = '{"gain": 12}'
        self.fs = BoardShim.get_sampling_rate(board_id)
        self.rows = [BoardShim.get_timestamp_channel(board_id), *BoardShim.get_eeg_channels(board_id)[:8]]
        if len(self.rows) != 9:
            raise RuntimeError("Expected eight EEG channels.")
        self.board = BoardShim(board_id, params)
        self.board.prepare_session()
        try:
            self.board.start_stream()
            time.sleep(2)
            for ch in range(1, 9):
                self.board.config_board(f"chon_{ch}_12")
                time.sleep(1)
                self.board.config_board(f"rldadd_{ch}")
                time.sleep(1)
            self.board.get_board_data()
        except BaseException:
            self.close()
            raise

    def read(self):
        if self.board:
            return self.board.get_board_data()[self.rows].T
        n = int((time.time() - self.last)*self.fs)
        t = self.last + np.arange(1, n+1)/self.fs
        self.last += n/self.fs
        # Clearly labelled simulated EEG, not a physiological recording.
        age = t - self.start_time
        transition = np.clip((age - 30.0) / 20.0, 0.0, 1.0) if self.demo_fatigue else np.zeros(n)
        theta = (2.0 + 12.0 * transition)[:, None] * np.sin(2*np.pi*6*t[:, None])
        alpha = (8.0 + 6.0 * transition)[:, None] * np.sin(2*np.pi*10*t[:, None])
        beta = (5.0 - 3.0 * transition)[:, None] * np.sin(2*np.pi*18*t[:, None])
        x = self.rng.normal(0, 2, (n, 8)) + theta + alpha + beta
        return np.column_stack((t, x))

    def mark(self, value):
        if self.board:
            self.board.insert_marker(value)

    def close(self):
        if self.board and self.board.is_prepared():
            try:
                self.board.stop_stream()
            finally:
                self.board.release_session()
