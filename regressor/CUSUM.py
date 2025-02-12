import numpy as np


def cusum_batch(signals, min_seg_len=10):
    """
    Apply the CUSUM approach to a batch of 2-channel signals for change-point detection
    and amplitude estimation.

    For each signal, the algorithm:
      1. Computes the overall mean for each channel.
      2. Computes the cumulative sum of deviations from the mean.
      3. Combines the two channels’ CUSUM statistics via the Euclidean norm.
      4. Searches for the candidate index (restricted by min_seg_len) at which the combined
         CUSUM statistic is maximized. This index is chosen as the jump index.
      5. Estimates the amplitudes as the means of the two segments (before and after the jump).

    Parameters:
        signals (np.ndarray): Array of shape (batch, 2, L) containing the signals.
        min_seg_len (int): Minimum allowed number of samples per segment. Default is 10.

    Returns:
        start_amplitudes (np.ndarray): Estimated start amplitudes, shape (batch, 2).
        stop_amplitudes (np.ndarray): Estimated stop amplitudes, shape (batch, 2).
        jump_indices (np.ndarray): Estimated jump indices, shape (batch,).
    """
    batch_size, num_channels, L = signals.shape

    # Prepare output arrays.
    start_amplitudes = np.zeros((batch_size, num_channels))
    stop_amplitudes = np.zeros((batch_size, num_channels))
    jump_indices = np.zeros(batch_size, dtype=int)

    for b in range(batch_size):
        signal = signals[b]  # shape: (2, L)

        # Compute overall mean per channel.
        mean_vals = np.mean(signal, axis=1)  # shape: (2,)

        # Compute cumulative sum deviations for each channel.
        # For channel 0:
        S0 = np.cumsum(signal[0] - mean_vals[0])
        # For channel 1:
        S1 = np.cumsum(signal[1] - mean_vals[1])

        # Combine the two channels' CUSUM statistics via the Euclidean norm.
        # This gives a single statistic per time step.
        S_comb = np.sqrt(S0**2 + S1**2)  # shape: (L,)

        # To avoid selecting a jump index too close to the boundaries,
        # we restrict candidate indices to [min_seg_len, L - min_seg_len).
        candidates = np.arange(L)
        valid_mask = (candidates >= min_seg_len) & (candidates < L - min_seg_len)
        valid_indices = candidates[valid_mask]
        valid_S = S_comb[valid_mask]

        # Select the candidate index where the combined CUSUM is maximized.
        best_candidate = valid_indices[np.argmax(valid_S)]
        jump_indices[b] = best_candidate

        # With the jump index determined, estimate the segment means (amplitudes) for each channel.
        for c in range(num_channels):
            start_amplitudes[b, c] = np.mean(signal[c, :best_candidate])
            stop_amplitudes[b, c] = np.mean(signal[c, best_candidate:])

    return start_amplitudes, stop_amplitudes, jump_indices


# === Example Usage ===
if __name__ == "__main__":
    np.random.seed(0)
    # Suppose we have a batch of 3 signals, each of length 512 with 2 channels.
    batch = 3
    L = 1024
    signals = np.zeros((batch, 2, L))

    # Define "true" parameters for synthetic signals:
    # For each signal, choose a jump index and start/stop amplitudes.
    true_jump_indices = np.array([200, 300, 150])
    true_start_amplitudes = np.array([[100, -50], [200, 0], [-300, 400]])
    true_stop_amplitudes = np.array([[150, -30], [250, 20], [-250, 450]])
    noise_std = 1300  # Gaussian noise standard deviation

    # Construct the signals and add noise.
    for b in range(batch):
        j = true_jump_indices[b]
        # Channel 0: in-phase; Channel 1: quadrature.
        signals[b, 0, :j] = true_start_amplitudes[b, 0]
        signals[b, 0, j:] = true_stop_amplitudes[b, 0]
        signals[b, 1, :j] = true_start_amplitudes[b, 1]
        signals[b, 1, j:] = true_stop_amplitudes[b, 1]
        # Add Gaussian noise to both channels.
        signals[b] += np.random.randn(2, L) * noise_std

    # Run the CUSUM-based change-point detection algorithm.
    est_start, est_stop, est_jumps = cusum_batch(signals, min_seg_len=2)

    print("Estimated start amplitudes:\n", est_start)
    print("Estimated stop amplitudes:\n", est_stop)
    print("Estimated jump indices:\n", est_jumps)
