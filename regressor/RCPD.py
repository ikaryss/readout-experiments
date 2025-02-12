import numpy as np
import matplotlib.pyplot as plt


def randomized_mean_window_cp(signal, window_size, num_iterations, diff_threshold=None):
    """
    Detects a change point in a two-channel piecewise constant signal (with at most one jump)
    using a randomized mean window algorithm.

    Parameters:
      signal : np.array, shape (N, 2)
          The two-channel signal (columns: in-phase and quadrature).
      window_size : int
          Size (in number of samples) of the window used for computing means.
      num_iterations : int
          Number of random windows to sample.
      diff_threshold : float, optional
          A threshold on the norm of the difference between the first and second half of
          a window. If not provided, it is set automatically as 50% of the maximum observed
          diff norm.

    Returns:
      estimated_jump : int or None
          Estimated sample index of the jump (None if no jump is detected).
      start_amp : np.array, shape (2,)
          Estimated amplitude (in-phase and quadrature) before the jump.
      stop_amp : np.array, shape (2,)
          Estimated amplitude (in-phase and quadrature) after the jump.
      details : dict
          Dictionary with intermediate data (centers, computed means, diff norms, etc.)
          that may be useful for diagnostics.
    """
    N = signal.shape[0]
    centers = []
    overall_means = []
    first_half_means = []
    second_half_means = []
    diffs = []
    diff_norms = []
    start_indices = []

    # Randomly sample windows
    for _ in range(num_iterations):
        # Choose a random start index such that the window fits inside the signal.
        start = np.random.randint(0, N - window_size + 1)
        end = start + window_size
        # Use the window's center as a candidate location for the jump.
        center = start + window_size // 2
        window = signal[start:end]

        # Compute overall mean and the means of the first and second halves
        overall_mean = np.mean(window, axis=0)
        half = window_size // 2
        first_half_mean = np.mean(window[:half], axis=0)
        second_half_mean = np.mean(window[half:], axis=0)
        diff = second_half_mean - first_half_mean
        diff_norm = np.linalg.norm(diff)

        centers.append(center)
        overall_means.append(overall_mean)
        first_half_means.append(first_half_mean)
        second_half_means.append(second_half_mean)
        diffs.append(diff)
        diff_norms.append(diff_norm)
        start_indices.append(start)

    # Convert lists to arrays for easier processing.
    centers = np.array(centers)
    overall_means = np.array(overall_means)  # shape (num_iterations, 2)
    first_half_means = np.array(first_half_means)
    second_half_means = np.array(second_half_means)
    diff_norms = np.array(diff_norms)
    diffs = np.array(diffs)
    start_indices = np.array(start_indices)

    # If no threshold was provided, set it automatically.
    # In windows that straddle a jump, diff_norm will be larger.
    if diff_threshold is None:
        diff_threshold = 0.5 * diff_norms.max()

    # Identify windows that likely straddle a change point.
    straddle_mask = diff_norms > diff_threshold

    if np.sum(straddle_mask) == 0:
        # If no window shows a significant difference between halves, assume no jump.
        estimated_jump = None
        # Use the overall median of all window means as the amplitude.
        amplitude_estimate = np.median(overall_means, axis=0)
        start_amp = amplitude_estimate
        stop_amp = amplitude_estimate
    else:
        # Use the centers of the straddling windows to estimate the jump location.
        estimated_jump = int(np.median(centers[straddle_mask]))

        # Now separate the windows into those clearly on either side of the candidate jump.
        before_mask = centers < estimated_jump
        after_mask = centers >= estimated_jump

        if np.sum(before_mask) > 0:
            start_amp = np.median(overall_means[before_mask], axis=0)
        else:
            start_amp = None

        if np.sum(after_mask) > 0:
            stop_amp = np.median(overall_means[after_mask], axis=0)
        else:
            stop_amp = None

    # Collect details for diagnostic purposes.
    details = {
        "centers": centers,
        "overall_means": overall_means,
        "first_half_means": first_half_means,
        "second_half_means": second_half_means,
        "diffs": diffs,
        "diff_norms": diff_norms,
        "straddle_mask": straddle_mask,
        "start_indices": start_indices,
        "diff_threshold": diff_threshold,
    }

    return estimated_jump, start_amp, stop_amp, details


# -------------------------------
# Example usage with synthetic data
# -------------------------------
if __name__ == "__main__":
    # For reproducibility.
    np.random.seed(0)

    # Create a synthetic two-channel signal.
    N = 1024
    jump_index = 50  # The true jump location.
    start_amp_true = np.array([-100, 400])  # True amplitude before the jump.
    stop_amp_true = np.array([-250, 450])  # True amplitude after the jump.
    noise_std = 10000

    # Construct the piecewise constant signal.
    signal = np.zeros((N, 2))
    signal[:jump_index] = start_amp_true
    signal[jump_index:] = stop_amp_true
    # Add white (Gaussian) noise.
    signal += np.random.randn(N, 2) * noise_std

    # Set parameters for the randomized window method.
    window_size = 30
    num_iterations = 1000

    # Run the change-point detection.
    est_jump, est_start_amp, est_stop_amp, details = randomized_mean_window_cp(
        signal, window_size, num_iterations
    )

    print("Estimated jump index:", est_jump)
    print("Estimated start amplitude:", est_start_amp)
    print("Estimated stop amplitude:", est_stop_amp)

    # -------------------------------
    # Diagnostic plot
    # -------------------------------
    # Plot the norm of the difference (between first and second halves of each window)
    # versus the window center. The straddling windows (above the threshold) are highlighted.
    plt.figure(figsize=(10, 6))
    plt.plot(
        details["centers"], details["diff_norms"], "o", alpha=0.3, label="Diff norm"
    )
    plt.axhline(
        details["diff_threshold"], color="red", linestyle="--", label="Threshold"
    )
    if est_jump is not None:
        plt.axvline(est_jump, color="green", linestyle="--", label="Estimated Jump")
    plt.xlabel("Window center index")
    plt.ylabel("Difference norm")
    plt.title("Randomized Mean Window Change-Point Detection")
    plt.legend()
    plt.show()
