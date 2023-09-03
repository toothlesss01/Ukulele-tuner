import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Constants for audio processing
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sample rate (Hz)
CHUNK_SIZE = 1024  # Number of frames per buffer
THRESHOLD = 7000  # Adjust this threshold for peak detection
TARGET_FREQS = {
    'G4': 392.0,
    'C4': 261.63,
    'E4': 329.63,
    'A4': 440.0,
}

def plot_spectrogram(data):
    plt.specgram(data, NFFT=1024, Fs=RATE, noverlap=512, cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Ukulele Tuner Spectrogram')
    plt.colorbar().set_label('Intensity (dB)')
    plt.show()

def tune_ukulele(data):
    freqs, amps = np.fft.fft(data), np.abs(np.fft.fft(data))
    peaks, _ = find_peaks(amps, height=THRESHOLD)

    if len(peaks) == 0:
        return "No peaks found, unable to determine tuningi."

    # Find the closest target frequency to the dominant peak
    dominant_freq = freqs[peaks[0]] 
    closest_note = min(TARGET_FREQS, key=lambda x: abs(TARGET_FREQS[x] - dominant_freq))

    if dominant_freq > TARGET_FREQS[closest_note]:
        return f"Tune down to {closest_note}"
    elif dominant_freq < TARGET_FREQS[closest_note]:
        return f"Tune up to {closest_note}"
    else:
        return f"In tune with {closest_note}"

def main():
    audio = pyaudio.PyAudio() 
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK_SIZE)

    print("Ukulele Tuner is listening...")

    try:
        while True:
            data = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
            plot_spectrogram(data)
            tuning_result = tune_ukulele(data)
            print(f"Tuning result: {tuning_result}")

    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()
