"""Phase space data generation with relaxation events"""

from typing import Tuple, List, Dict, Optional, NamedTuple
import numpy as np
from dataclasses import dataclass


from typing import Union, Sequence

AmplitudeType = Union[float, Sequence[float]]


class IQParameters(NamedTuple):
    """
    Parameters for IQ data generation

    Parameters
    ----------
    I_amp : float or sequence of float
        In-phase amplitude. If sequence of length 2, uniformly samples between values
    Q_amp : float or sequence of float
        Quadrature amplitude. If sequence of length 2, uniformly samples between values
    """

    I_amp: AmplitudeType
    Q_amp: AmplitudeType

    def sample_amplitude(self, amp: AmplitudeType) -> float:
        """Sample amplitude value, either returning fixed value or sampling from range."""
        if isinstance(amp, (int, float)):
            return float(amp)
        if len(amp) != 2:
            raise ValueError("Amplitude range must be sequence of length 2")
        return float(np.random.uniform(amp[0], amp[1]))


@dataclass
class RelaxationParameters:
    """Parameters for relaxation simulation"""

    T1: float
    relax_time_transition: float


class QuantumStateGenerator:
    """Generator for quantum state measurement data in phase space.

    This class provides methods to generate synthetic measurement data for different
    quantum states and scenarios, including ground state, excited state, and
    relaxation events.
    """

    def __init__(
        self,
        meas_time: np.ndarray,
        excited_params: Optional[IQParameters] = None,
        ground_params: Optional[IQParameters] = None,
        relaxation_params: Optional[RelaxationParameters] = None,
        gauss_noise_amp: float = 0.0,
        qubit: int = 0,
    ):
        """
        Initialize the quantum state generator.

        Parameters
        ----------
        meas_time : np.ndarray
            Time points for measurements
        excited_params : IQParameters, optional
            IQ parameters for excited state
        ground_params : IQParameters, optional
            IQ parameters for ground state
        relaxation_params : RelaxationParameters, optional
            Parameters for relaxation simulation
        gauss_noise_amp : float, default=0.0
            Amplitude of Gaussian noise to add
        qubit : int, default=0
            Qubit identifier for labeling
        """
        self.meas_time = meas_time
        self.excited_params = excited_params
        self.ground_params = ground_params
        self.relaxation_params = relaxation_params
        self.gauss_noise_amp = gauss_noise_amp
        self.qubit = qubit
        self.dt = meas_time[1] - meas_time[0]

    def _add_gaussian_noise(
        self, I_values: np.ndarray, Q_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add Gaussian noise to I and Q values if noise amplitude is set."""
        if self.gauss_noise_amp > 0:
            I_noise = np.random.normal(
                0, self.gauss_noise_amp, size=len(self.meas_time)
            )
            Q_noise = np.random.normal(
                0, self.gauss_noise_amp, size=len(self.meas_time)
            )
            return I_values + I_noise, Q_values + Q_noise
        return I_values, Q_values

    def _generate_constant_IQ(self, params: IQParameters) -> np.ndarray:
        """
        Generate constant IQ values with optional noise.

        If I_amp or Q_amp is a sequence of length 2, uniformly samples amplitude
        from the specified range.
        """
        # Sample amplitudes if ranges provided
        I_amp = params.sample_amplitude(params.I_amp)
        Q_amp = params.sample_amplitude(params.Q_amp)

        I_values = np.ones_like(self.meas_time) * I_amp
        Q_values = np.ones_like(self.meas_time) * Q_amp
        I_values, Q_values = self._add_gaussian_noise(I_values, Q_values)
        return I_values + 1j * Q_values

    def generate_ground_state(
        self, batch_size: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[int, int]]]:
        """
        Generate ground state (|0⟩) measurement data.

        Parameters
        ----------
        batch_size : int
            Number of experiments to generate
        seed : int, optional
            Random seed for reproducible generation

        Returns
        -------
        Tuple[np.ndarray, List[Dict[int, int]]]
            Complex-valued array of shape (batch_size, len(meas_time)) and labels
        """
        if self.ground_params is None:
            raise ValueError("Ground state parameters not provided")

        if seed is not None:
            np.random.seed(seed)

        data = np.zeros((batch_size, len(self.meas_time)), dtype=np.complex64)
        labels = []

        for i in range(batch_size):
            data[i] = self._generate_constant_IQ(self.ground_params)
            labels.append({self.qubit: 0})

        return data, labels

    def generate_excited_state(
        self, batch_size: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[int, int]]]:
        """
        Generate excited state (|1⟩) measurement data.

        Parameters
        ----------
        batch_size : int
            Number of experiments to generate
        seed : int, optional
            Random seed for reproducible generation

        Returns
        -------
        Tuple[np.ndarray, List[Dict[int, int]]]
            Complex-valued array of shape (batch_size, len(meas_time)) and labels
        """
        if self.excited_params is None:
            raise ValueError("Excited state parameters not provided")

        if seed is not None:
            np.random.seed(seed)

        data = np.zeros((batch_size, len(self.meas_time)), dtype=np.complex64)
        labels = []

        for i in range(batch_size):
            data[i] = self._generate_constant_IQ(self.excited_params)
            labels.append({self.qubit: 1})

        return data, labels

    def generate_relaxation_event(
        self,
        batch_size: int,
        uniform_sampling: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[int, int]]]:
        """
        Generate measurement data with guaranteed relaxation events.

        Parameters
        ----------
        batch_size : int
            Number of experiments to generate
        uniform_sampling : bool, default=False
            If True, samples relaxation times from uniform distribution.
            If False, samples based on T1 exponential decay probability distribution.
        seed : int, optional
            Random seed for reproducible generation

        Returns
        -------
        Tuple[np.ndarray, List[Dict[int, int]]]
            Complex-valued array of shape (batch_size, len(meas_time)) and labels
        """
        if any(
            param is None
            for param in [
                self.excited_params,
                self.ground_params,
                self.relaxation_params,
            ]
        ):
            raise ValueError("Missing required parameters for relaxation simulation")

        if seed is not None:
            np.random.seed(seed)

        data = np.zeros((batch_size, len(self.meas_time)), dtype=np.complex64)
        labels = []

        for i in range(batch_size):
            if uniform_sampling:
                # Sample from uniform distribution
                relax_idx = np.random.randint(0, len(self.meas_time) - 1)
            else:
                # Sample based on T1 relaxation probability
                relax_probs = 1 - np.exp(-self.meas_time / self.relaxation_params.T1)
                # Normalize probabilities to form a valid distribution
                relax_probs = relax_probs / np.sum(relax_probs)
                # Sample index based on probability distribution
                relax_idx = np.random.choice(len(self.meas_time), p=relax_probs)
            data[i] = self._generate_relaxation_trajectory(relax_idx)
            labels.append({self.qubit: 2})

        return data, labels

    def _generate_relaxation_trajectory(self, relax_idx: int) -> np.ndarray:
        """Generate a single relaxation trajectory starting at given index."""
        # Sample initial amplitudes
        excited_I = self.excited_params.sample_amplitude(self.excited_params.I_amp)
        excited_Q = self.excited_params.sample_amplitude(self.excited_params.Q_amp)
        ground_I = self.ground_params.sample_amplitude(self.ground_params.I_amp)
        ground_Q = self.ground_params.sample_amplitude(self.ground_params.Q_amp)

        # Initialize with excited state amplitudes
        I_values = np.ones_like(self.meas_time) * excited_I
        Q_values = np.ones_like(self.meas_time) * excited_Q

        # Calculate transition window
        transition_points = max(
            1, int(self.relaxation_params.relax_time_transition / self.dt)
        )
        transition_window = np.linspace(0, 1, transition_points)

        # Apply transition
        start_idx = relax_idx
        end_idx = min(start_idx + transition_points, len(self.meas_time))
        window_size = end_idx - start_idx

        I_values[start_idx:end_idx] = (
            excited_I + (ground_I - excited_I) * transition_window[:window_size]
        )
        Q_values[start_idx:end_idx] = (
            excited_Q + (ground_Q - excited_Q) * transition_window[:window_size]
        )

        if end_idx < len(self.meas_time):
            I_values[end_idx:] = ground_I
            Q_values[end_idx:] = ground_Q

        I_values, Q_values = self._add_gaussian_noise(I_values, Q_values)
        return I_values + 1j * Q_values

    def simulate_measurement(
        self, batch_size: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict[int, int]]]:
        """
        Simulate realistic measurement behavior with possible relaxation events.

        This simulates the behavior of preparing the excited state and measuring,
        where relaxation events may occur based on T1 time constant.

        Parameters
        ----------
        batch_size : int
            Number of experiments to generate
        seed : int, optional
            Random seed for reproducible generation

        Returns
        -------
        Tuple[np.ndarray, List[Dict[int, int]]]
            Complex-valued array of shape (batch_size, len(meas_time)) and labels
        """
        if any(
            param is None
            for param in [
                self.excited_params,
                self.ground_params,
                self.relaxation_params,
            ]
        ):
            raise ValueError("Missing required parameters for measurement simulation")

        if seed is not None:
            np.random.seed(seed)

        data = np.zeros((batch_size, len(self.meas_time)), dtype=np.complex64)
        labels = []

        for i in range(batch_size):
            # Calculate relaxation probability based on T1
            rand_val = np.random.random()
            # Probability of relaxation is 1 - exp(-t/T1)
            relax_probs = 1 - np.exp(-self.meas_time / self.relaxation_params.T1)
            # Find the first time point where relaxation probability exceeds random value
            relax_idx = np.searchsorted(relax_probs, rand_val)

            if relax_idx < len(self.meas_time):
                data[i] = self._generate_relaxation_trajectory(relax_idx)
                labels.append({self.qubit: 2})
            else:
                data[i] = self._generate_constant_IQ(self.excited_params)
                labels.append({self.qubit: 1})

        return data, labels


if __name__ == "__main__":
    # Example usage demonstrating all generation modes
    meas_time = np.linspace(0, 100e-6, 1000)  # 100 microseconds, 1000 points

    # Define parameters with amplitude ranges
    excited = IQParameters(I_amp=[0.8, 1.2], Q_amp=[0.4, 0.6])  # Sample from ranges
    ground = IQParameters(I_amp=0.0, Q_amp=0.0)  # Fixed values
    relaxation = RelaxationParameters(T1=20e-6, relax_time_transition=1e-6)

    # Create generator instance
    generator = QuantumStateGenerator(
        meas_time=meas_time,
        excited_params=excited,
        ground_params=ground,
        relaxation_params=relaxation,
        gauss_noise_amp=0.1,
        qubit=0,
    )

    # Generate ground state data
    data0, labels0 = generator.generate_ground_state(batch_size=10, seed=42)
    print("\nGround State Generation:")
    print(f"Data shape: {data0.shape}")
    print(f"First few labels: {labels0[:3]}")

    # Generate excited state data
    data1, labels1 = generator.generate_excited_state(batch_size=10, seed=42)
    print("\nExcited State Generation:")
    print(f"Data shape: {data1.shape}")
    print(f"First few labels: {labels1[:3]}")

    # Generate relaxation events with uniform sampling
    data2_uniform, labels2_uniform = generator.generate_relaxation_event(
        batch_size=10, uniform_sampling=True, seed=42
    )
    print("\nRelaxation Events Generation (Uniform Sampling):")
    print(f"Data shape: {data2_uniform.shape}")
    print(f"First few labels: {labels2_uniform[:3]}")

    # Generate relaxation events with T1-based sampling
    data2_t1, labels2_t1 = generator.generate_relaxation_event(
        batch_size=10, uniform_sampling=False, seed=42
    )
    print("\nRelaxation Events Generation (T1-based Sampling):")
    print(f"Data shape: {data2_t1.shape}")
    print(f"First few labels: {labels2_t1[:3]}")

    # Simulate realistic measurement
    data3, labels3 = generator.simulate_measurement(batch_size=10, seed=42)
    print("\nMeasurement Simulation:")
    print(f"Data shape: {data3.shape}")
    print(f"First few labels: {labels3[:3]}")

    # Demonstrate reproducibility
    data3_repeat, labels3_repeat = generator.simulate_measurement(
        batch_size=10, seed=42
    )
    print("\nReproducibility check:")
    print(f"Data is identical: {np.allclose(data3, data3_repeat)}")
    print(f"Labels are identical: {labels3 == labels3_repeat}")
