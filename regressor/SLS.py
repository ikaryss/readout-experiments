import numpy as np


def segmented_least_squares_batch(signals, min_seg_len=1):
    """
    Apply segmented least squares / exhaustive search to a batch of 2-channel signals.

    For each signal, the algorithm searches over candidate jump indices (j) such that:
      - Left segment: samples [0, j-1]
      - Right segment: samples [j, L-1]
    and computes the sum of squared errors (SSE) for both segments (per channel).
    The candidate jump that minimizes the total SSE (summed over the two channels)
    is chosen as the change point. The estimated amplitudes are the mean values on
    the left and right segments.

    Parameters:
        signals (np.ndarray): Array of shape (batch, 2, L) containing the signals.
        min_seg_len (int): Minimum allowed number of samples per segment.
                           (Default is 1; you might set this to a higher value to avoid
                           very short segments.)

    Returns:
        start_amplitudes (np.ndarray): Estimated start amplitudes, shape (batch, 2)
        stop_amplitudes (np.ndarray): Estimated stop amplitudes, shape (batch, 2)
        jump_indices (np.ndarray): Estimated jump indices (first sample of second segment), shape (batch,)
    """
    batch_size, num_channels, L = signals.shape

    # Prepare arrays to store the estimates
    start_amplitudes = np.zeros((batch_size, num_channels))
    stop_amplitudes = np.zeros((batch_size, num_channels))
    jump_indices = np.zeros(batch_size, dtype=int)

    # Define the valid candidate jump indices.
    # We require the left segment to have at least min_seg_len samples and similarly for the right.
    valid_js = np.arange(
        min_seg_len, L - min_seg_len + 1
    )  # candidate j: left segment has j samples

    # Process each signal in the batch
    for b in range(batch_size):
        signal = signals[b]  # shape: (2, L)
        # We will compute the SSE for each candidate jump for each channel.
        # Then we sum the SSE across channels.
        sse_channels = (
            []
        )  # to store SSE for each channel (each as an array over candidate j's)

        for c in range(num_channels):
            x = signal[c]  # shape (L,)
            # Compute cumulative sum and cumulative sum of squares
            # Prepending 0 makes it easy to compute sums over any segment.
            cumsum = np.concatenate(([0], np.cumsum(x)))  # shape: (L+1,)
            cumsum2 = np.concatenate(([0], np.cumsum(x**2)))  # shape: (L+1,)

            # For each candidate jump j in valid_js:
            # Left segment: indices [0, j-1] (length = j)
            # Right segment: indices [j, L-1] (length = L - j)
            j = valid_js  # array of candidate j values

            # Compute SSE for the left segment.
            # The sum over the left segment is cumsum[j] and its length is j.
            # The mean over left segment is cumsum[j] / j.
            # The sum of squared errors (SSE) for the left segment is:
            #   SSE_left = cumsum2[j] - (cumsum[j]**2)/j
            SSE_left = cumsum2[j] - (cumsum[j] ** 2) / j

            # Compute SSE for the right segment.
            # Sum over right segment: cumsum[L] - cumsum[j], length: L - j.
            # Mean over right segment: (cumsum[L] - cumsum[j])/(L - j)
            # SSE_right = (cumsum2[L] - cumsum2[j]) - ((cumsum[L]-cumsum[j])**2)/(L - j)
            SSE_right = (cumsum2[L] - cumsum2[j]) - ((cumsum[L] - cumsum[j]) ** 2) / (
                L - j
            )

            # Total SSE for channel c for each candidate j:
            sse_c = SSE_left + SSE_right
            sse_channels.append(sse_c)

        # Sum the SSEs across the two channels (axis=0 corresponds to candidate j)
        sse_total = np.sum(sse_channels, axis=0)

        # Find the candidate jump index (from valid_js) that minimizes the total SSE.
        best_candidate_idx = np.argmin(sse_total)
        best_j = valid_js[best_candidate_idx]
        jump_indices[b] = best_j

        # Given the chosen jump index, compute the estimated amplitudes for each channel.
        for c in range(num_channels):
            x = signal[c]
            start_amplitudes[b, c] = np.mean(x[:best_j])
            stop_amplitudes[b, c] = np.mean(x[best_j:])

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
        # Add Gaussian noise
        signals[b] += np.random.randn(2, L) * noise_std

    # Run the segmented least squares algorithm.
    # Here we set min_seg_len to 100 to avoid very short segments.
    est_start, est_stop, est_jumps = segmented_least_squares_batch(
        signals, min_seg_len=5
    )

    print("Estimated start amplitudes:\n", est_start)
    print("Estimated stop amplitudes:\n", est_stop)
    print("Estimated jump indices:\n", est_jumps)
