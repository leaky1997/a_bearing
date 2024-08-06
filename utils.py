import numpy as np
import matplotlib.pyplot as plt

def create_overlapping_dataset(data, window_size, overlap_percent):
    step_size = int(window_size * (1 - overlap_percent))
    result = []
    for i in range(0, len(data) - window_size + 1, step_size):
        result.append(data[i:i+window_size])
        # plt.plot(result[-1][:,2])
        # plt.show()
    return np.array(result)

def compute_frequency_domain(signal, sample_rate):
    n = len(signal)
    fft_values = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, 1.0/sample_rate)
    return frequencies, np.abs(fft_values)

def plot_time_and_frequency(data, sample_rate,name,fre_range = 300):
    plt.figure(figsize=(12, 8))

    for i in range(data.shape[1]):
        signal = data[:, i]
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        plt.subplot(2, 2, i+1)
        plt.plot(signal, label = f'Channel {i+1}',c = 'purple')
        plt.title(f'Time Domain - Channel {i+1}')

        frequencies, fft_values = compute_frequency_domain(signal, sample_rate)
        plt.subplot(2, 2, i+3)
        plt.plot(frequencies[1:fre_range], fft_values[1:fre_range], c = 'darkblue')
        plt.title(f'Frequency Domain - Channel {i+1}')

    plt.tight_layout()
    plt.savefig(f'save/{name}.pdf')
    plt.show()