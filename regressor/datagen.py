# datagen.py
import numpy as np
import torch


class DataGenerator:
    """
    DataGenerator generates synthetic (X, y) training examples for a change-point detection problem.

    Each sample is a two-channel (in-phase and quadrature) time series of fixed length.
    The 'ground truth' y is provided as a tuple:
        (amplitude_target, cp_target)
      - For constant signals, amplitude_target is a 4-dimensional vector where
        [amp_I, amp_Q, amp_I, amp_Q] (i.e. start and stop amplitudes are identical),
        and cp_target is set to -1.0 to indicate no jump.
      - For signals with a jump, amplitude_target is a 4-dimensional vector
        [amp_start_I, amp_start_Q, amp_stop_I, amp_stop_Q] and cp_target is the
        (randomly chosen) sample index where the jump occurs (the same for both channels).

    The IQ amplitudes are sampled uniformly from a specified range,
    and the Gaussian noise standard deviation is sampled uniformly from a specified range.
    """

    def __init__(
        self,
        signal_length=100,
        amplitude_range=(-1000, 1000),
        noise_std_range=(1300, 1500),
        seed=None,
    ):
        """
        Initialize the data generator.

        Args:
            signal_length (int): Number of time steps in each signal.
            amplitude_range (tuple): (min_amplitude, max_amplitude) for IQ signal amplitudes.
            noise_std_range (tuple): (min_std, max_std) for noise standard deviation.
            seed (int, optional): Random seed for reproducibility.
        """
        self.signal_length = signal_length
        self.amplitude_range = amplitude_range
        self.noise_std_range = noise_std_range
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate_constant(self, batch_size):
        """
        Generate a batch of samples with constant IQ values (i.e. no jump).

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            X (torch.Tensor): Tensor of shape (batch_size, 2, signal_length) with the noisy signal.
            y (tuple): Tuple (amplitude_target, cp_target) where:
                   - amplitude_target is a tensor of shape (batch_size, 4) in the order
                     [amp_I, amp_Q, amp_I, amp_Q].
                   - cp_target is a tensor of shape (batch_size,) with a value -1.0 for each sample.
        """
        T = self.signal_length
        X_list = []
        amp_targets = []
        cp_targets = []
        for _ in range(batch_size):
            # Sample constant amplitudes for in-phase and quadrature channels
            amp_I = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
            amp_Q = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])

            # Create constant signal for both channels
            signal = np.array([[amp_I] * T, [amp_Q] * T], dtype=np.float32)

            # Sample a noise standard deviation uniformly
            noise_std = np.random.uniform(
                self.noise_std_range[0], self.noise_std_range[1]
            )
            noise = np.random.randn(2, T).astype(np.float32) * noise_std

            # Final signal is the sum of the signal and noise
            X_sample = signal + noise
            X_list.append(X_sample)

            # Ground truth amplitudes: both start and stop are the same
            amp_target = np.array([amp_I, amp_Q, amp_I, amp_Q], dtype=np.float32)
            amp_targets.append(amp_target)

            # For constant signal, no jump; set jump target to -1 (or any flag)
            cp_targets.append(-1.0)

        # Convert lists to torch tensors
        X_tensor = torch.tensor(np.array(X_list))  # Shape: (batch_size, 2, T)
        amp_tensor = torch.tensor(np.array(amp_targets))  # Shape: (batch_size, 4)
        cp_tensor = torch.tensor(np.array(cp_targets))  # Shape: (batch_size,)
        return X_tensor, (amp_tensor, cp_tensor)

    def generate_jump(self, batch_size):
        """
        Generate a batch of samples with a jump (change-point) occurring at the same sample in both channels.

        The jump index is sampled uniformly from 2 to signal_length - 1.

        Args:
            batch_size (int): Number of samples to generate.

        Returns:
            X (torch.Tensor): Tensor of shape (batch_size, 2, signal_length) with the noisy signal.
            y (tuple): Tuple (amplitude_target, cp_target) where:
                   - amplitude_target is a tensor of shape (batch_size, 4) in the order
                     [amp_start_I, amp_start_Q, amp_stop_I, amp_stop_Q].
                   - cp_target is a tensor of shape (batch_size,) with the jump index.
        """
        T = self.signal_length
        X_list = []
        amp_targets = []
        cp_targets = []
        for _ in range(batch_size):
            # Randomly choose a jump index (avoid boundaries)
            cp = np.random.randint(2, T - 1)

            # Sample amplitudes for before jump (start) and after jump (stop) for both channels
            amp_start_I = np.random.uniform(
                self.amplitude_range[0], self.amplitude_range[1]
            )
            amp_start_Q = np.random.uniform(
                self.amplitude_range[0], self.amplitude_range[1]
            )
            amp_stop_I = np.random.uniform(
                self.amplitude_range[0], self.amplitude_range[1]
            )
            amp_stop_Q = np.random.uniform(
                self.amplitude_range[0], self.amplitude_range[1]
            )

            # Create a piecewise constant signal with the jump
            signal = np.zeros((2, T), dtype=np.float32)
            # Before the jump: use start amplitudes
            signal[0, :cp] = amp_start_I
            signal[1, :cp] = amp_start_Q
            # After (and including) the jump: use stop amplitudes
            signal[0, cp:] = amp_stop_I
            signal[1, cp:] = amp_stop_Q

            # Sample noise standard deviation uniformly
            noise_std = np.random.uniform(
                self.noise_std_range[0], self.noise_std_range[1]
            )
            noise = np.random.randn(2, T).astype(np.float32) * noise_std

            # Final signal with noise
            X_sample = signal + noise
            X_list.append(X_sample)

            # Ground truth amplitudes and jump index
            amp_target = np.array(
                [amp_start_I, amp_start_Q, amp_stop_I, amp_stop_Q], dtype=np.float32
            )
            amp_targets.append(amp_target)
            cp_targets.append(float(cp))

        X_tensor = torch.tensor(np.array(X_list))
        amp_tensor = torch.tensor(np.array(amp_targets))
        cp_tensor = torch.tensor(np.array(cp_targets))
        return X_tensor, (amp_tensor, cp_tensor)

    def generate_data(self, batch_size, jump=True):
        """
        Generate a batch of data.

        Args:
            batch_size (int): Number of samples.
            jump (bool): If True, generate data with a jump;
                         otherwise, generate constant IQ signals.

        Returns:
            X (torch.Tensor) and y (tuple) as defined in the above methods.
        """
        if jump:
            return self.generate_jump(batch_size)
        else:
            return self.generate_constant(batch_size)


# Example usage (for testing the module)
if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42

    # Create an instance of DataGenerator with desired parameters
    generator = DataGenerator(
        signal_length=512,
        amplitude_range=(-1000, 1000),
        noise_std_range=(1300, 1500),
        seed=seed,
    )

    # Generate constant IQ data
    X_const, y_const = generator.generate_data(batch_size=5, jump=False)
    print("Constant data X shape:", X_const.shape)
    print("Constant data y (amplitudes) shape:", y_const[0].shape)
    print("Constant data y (cp) shape:", y_const[1].shape)

    # Generate jump data
    X_jump, y_jump = generator.generate_data(batch_size=5, jump=True)
    print("Jump data X shape:", X_jump.shape)
    print("Jump data y (amplitudes) shape:", y_jump[0].shape)
    print("Jump data y (cp) shape:", y_jump[1].shape)
