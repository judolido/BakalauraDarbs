
import socket
import struct
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sounddevice as sd
import wave
from datetime import datetime

# ==== Config ====
UDP_PORT = 5004
UDP_IP = "0.0.0.0"   # listen on all interfaces
BYTES_PER_SAMPLE = 3
SAMPLES_PER_PKT = 480
HEADER_SIZE = 8
SAMPLE_RATE = 32000   # set to your actual FS (e.g., 32000 or 22050)
WINDOW_SIZE = 2048

MAGIC0 = 0xAA
MAGIC1 = 0x55

def bytes_to_signed_24bit(b: bytes) -> int:
    value = b[0] | (b[1] << 8) | (b[2] << 16)
    if value & 0x800000:
        value -= 1 << 24
    return value

class UDPReader(QtCore.QThread):
    data_received = QtCore.pyqtSignal(bytes)  # raw payload (PCM24 bytes)

    def __init__(self, ip="0.0.0.0", port=5004, parent=None):
        super().__init__(parent)
        self.ip = ip
        self.port = port
        self._running = True
        self.sock = None

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.ip, self.port))
            self.sock.settimeout(1.0)
            print(f"Listening UDP on {self.ip}:{self.port}")
        except Exception as e:
            print(f"Error opening UDP socket: {e}")
            return

        while self._running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if not data or len(data) < HEADER_SIZE:
                    continue
                if data[0] != MAGIC0 or data[1] != MAGIC1:
                    continue  # ignore non-audio packets
                # seq = data[2]
                # flags = data[3]
                # ts = struct.unpack_from('<I', data, 4)[0]
                payload = data[HEADER_SIZE:]
                if len(payload) % BYTES_PER_SAMPLE != 0:
                    continue
                self.data_received.emit(payload)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP read error: {e}")
                self._running = False

        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait()

class AudioVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Audio Visualizer (UDP)")

        # ==== Buffers ====
        self.data = np.zeros(WINDOW_SIZE, dtype=np.int32)
        self.plot_fft = False
        self.recording = False
        self.playing = False
        self.recorded_samples = []
        self.gain_factor = 1

        # ==== GUI ====
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot_widget = pg.PlotWidget(title="Audio Input")
        self.curve = self.plot_widget.plot(self.data, pen='y')
        self.plot_widget.setYRange(-8500000, 8500000)
        self.plot_widget.setXRange(0, WINDOW_SIZE)
        layout.addWidget(self.plot_widget)

        btn_layout = QtWidgets.QHBoxLayout()

        self.mode_btn = QtWidgets.QPushButton("Switch to FFT")
        self.mode_btn.clicked.connect(self.toggle_mode)
        btn_layout.addWidget(self.mode_btn)

        self.record_btn = QtWidgets.QPushButton("Start Recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        btn_layout.addWidget(self.record_btn)

        self.play_btn = QtWidgets.QPushButton("Play Recording")
        self.play_btn.clicked.connect(self.play_recording)
        btn_layout.addWidget(self.play_btn)

        self.idle_btn = QtWidgets.QPushButton("Idle Mode")
        self.idle_btn.clicked.connect(self.set_idle_mode)
        btn_layout.addWidget(self.idle_btn)

        # Gain controls
        self.gain_label = QtWidgets.QLabel(f"Gain: x{self.gain_factor}")
        btn_layout.addWidget(self.gain_label)

        self.gain_inc_btn = QtWidgets.QPushButton("Gain +")
        self.gain_inc_btn.clicked.connect(self.increase_gain)
        btn_layout.addWidget(self.gain_inc_btn)

        self.gain_dec_btn = QtWidgets.QPushButton("Gain -")
        self.gain_dec_btn.clicked.connect(self.decrease_gain)
        btn_layout.addWidget(self.gain_dec_btn)

        layout.addLayout(btn_layout)

        # ==== UDP reader thread ====
        self.net_thread = UDPReader(UDP_IP, UDP_PORT)
        self.net_thread.data_received.connect(self.on_network_data)
        self.net_thread.start()

        # ==== Update timer ====
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)  # ~50 FPS

    def increase_gain(self):
        self.gain_factor *= 2
        self.gain_label.setText(f"Gain: x{self.gain_factor}")

    def decrease_gain(self):
        if self.gain_factor > 1:
            self.gain_factor //= 2
            self.gain_label.setText(f"Gain: x{self.gain_factor}")

    def toggle_mode(self):
        self.plot_fft = not self.plot_fft
        self.mode_btn.setText("Switch to Waveform" if self.plot_fft else "Switch to FFT")
        if self.plot_fft:
            self.plot_widget.setYRange(0, 1e7)
            self.plot_widget.setTitle("FFT Spectrum")
            self.plot_widget.setXRange(0, SAMPLE_RATE / 2)
        else:
            self.plot_widget.setYRange(-8500000, 8500000)
            self.plot_widget.setTitle("Audio Waveform")

    def toggle_recording(self):
        self.recording = self.record_btn.isChecked()
        self.record_btn.setText("Stop Recording" if self.recording else "Start Recording")
        if not self.recording and self.recorded_samples:
            self.save_wav()

    def set_idle_mode(self):
        self.recording = False
        self.record_btn.setChecked(False)
        self.record_btn.setText("Start Recording")
        self.playing = False
        self.plot_widget.setTitle("Live Only (Idle Mode)")

    def play_recording(self):
        if not self.recorded_samples:
            print("No recording to play.")
            return
        self.playing = True
        print("Playing recording...")
        norm_data = np.array(self.recorded_samples, dtype=np.float32) / 8388608.0
        sd.play(norm_data, SAMPLE_RATE)
        sd.wait()
        self.playing = False
        print("Playback done.")

    def save_wav(self):
        filename = datetime.now().strftime("recorded_%Y%m%d_%H%M%S.wav")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)
            wf.setframerate(SAMPLE_RATE)
            for sample in self.recorded_samples:
                s = max(min(sample, 8388607), -8388608)
                wf.writeframes(struct.pack('<i', s)[:3])
        print(f"Saved: {filename}")

    @QtCore.pyqtSlot(bytes)
    def on_network_data(self, payload: bytes):
        # payload is N * 3 bytes of PCM24 little-endian
        n = len(payload) // BYTES_PER_SAMPLE
        if n == 0:
            return

        # Convert efficiently using numpy from bytes -> int32
        mv = memoryview(payload)
        samples = []
        for i in range(n):
            b = mv[i*3:(i+1)*3].tobytes()
            s = bytes_to_signed_24bit(b)
            s = int(s * self.gain_factor)
            s = max(min(s, 8388607), -8388608)
            samples.append(s)

        samples = np.array(samples, dtype=np.int32)

        if self.recording:
            self.recorded_samples.extend(samples.tolist())

        # Update ring buffer for plotting
        shift_len = min(len(samples), WINDOW_SIZE)
        self.data = np.roll(self.data, -shift_len)
        self.data[-shift_len:] = samples[-shift_len:]

    def update_plot(self):
        if self.plot_fft:
            fft_data = np.fft.rfft(self.data * np.hamming(len(self.data)))
            fft_magnitude = np.abs(fft_data)
            freqs = np.fft.rfftfreq(len(self.data), d=1.0 / SAMPLE_RATE)
            self.curve.setData(freqs, fft_magnitude)
        else:
            self.curve.setData(self.data)

    def closeEvent(self, event):
        self.net_thread.stop()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    visualizer = AudioVisualizer()
    visualizer.show()
    sys.exit(app.exec_())
