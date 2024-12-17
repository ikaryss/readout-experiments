"Phase space data generation with relaxation events"

from typing import Tuple, List, Dict, Optional
import numpy as np


# pylint: disable=invalid-name, redefined-outer-name


def generate_relaxation_data(
    batch_size: int,
    meas_time: np.ndarray,
    I_amp_1: Optional[float] = None,
    Q_amp_1: Optional[float] = None,
    I_amp_0: Optional[float] = None,
    Q_amp_0: Optional[float] = None,
    relax_time_transition: Optional[float] = None,
    T1: Optional[float] = None,
    gauss_noise_amp: float = 0.0,
    qubit: int = 0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[int, int]]]:
    """
    Generate synthetic qubit measurement data with optional relaxation events.

    The function has three modes of operation:
    1. If only I_amp_1, Q_amp_1 provided: generates data without relaxation (label 1)
    2. If only I_amp_0, Q_amp_0 provided: generates data in relaxed state (label 0)
    3. If all amplitude parameters and T1 provided: generates data with possible
       relaxation events (labels 1 or 2)

    Parameters
    ----------
    batch_size : int
        Number of experiments to generate
    meas_time : np.ndarray
        Time points for measurements
    I_amp_1 : float, optional
        Initial in-phase amplitude (before relaxation)
    Q_amp_1 : float, optional
        Initial quadrature amplitude (before relaxation)
    I_amp_0 : float, optional
        Final in-phase amplitude after relaxation
    Q_amp_0 : float, optional
        Final quadrature amplitude after relaxation
    relax_time_transition : float, optional
        Time constant for the relaxation transition
    T1 : float, optional
        Relaxation time constant
    gauss_noise_amp : float, default=0.0
        Amplitude of Gaussian noise to add
    qubit : int, default=0
        Qubit identifier for labeling
    seed : int, optional
        Random seed for reproducible data generation

    Returns
    -------
    Tuple[np.ndarray, List[Dict[int, int]]]
        - Complex-valued array of shape (batch_size, len(meas_time))
          containing the generated time series
        - List of dictionaries containing labels for each experiment:
          label=0: relaxed state (when only I_amp_0, Q_amp_0 provided)
          label=1: excited state without relaxation
          label=2: relaxation occurred during measurement

    Raises
    ------
    ValueError
        If the combination of provided parameters doesn't match any valid mode
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Initialize output array
    data = np.zeros((batch_size, len(meas_time)), dtype=np.complex64)
    labels = []

    # Determine operation mode based on provided parameters
    if (
        I_amp_1 is not None
        and Q_amp_1 is not None
        and I_amp_0 is None
        and Q_amp_0 is None
    ):
        # Mode 1: Generate data without relaxation (label 1)
        for i in range(batch_size):
            I_values = np.ones_like(meas_time) * I_amp_1
            Q_values = np.ones_like(meas_time) * Q_amp_1

            if gauss_noise_amp > 0:
                I_noise = np.random.normal(0, gauss_noise_amp, size=len(meas_time))
                Q_noise = np.random.normal(0, gauss_noise_amp, size=len(meas_time))
                I_values += I_noise
                Q_values += Q_noise

            data[i] = I_values + 1j * Q_values
            labels.append({qubit: 1})

    elif (
        I_amp_0 is not None
        and Q_amp_0 is not None
        and I_amp_1 is None
        and Q_amp_1 is None
    ):
        # Mode 2: Generate data in relaxed state (label 0)
        for i in range(batch_size):
            I_values = np.ones_like(meas_time) * I_amp_0
            Q_values = np.ones_like(meas_time) * Q_amp_0

            if gauss_noise_amp > 0:
                I_noise = np.random.normal(0, gauss_noise_amp, size=len(meas_time))
                Q_noise = np.random.normal(0, gauss_noise_amp, size=len(meas_time))
                I_values += I_noise
                Q_values += Q_noise

            data[i] = I_values + 1j * Q_values
            labels.append({qubit: 0})

    elif all(param is not None for param in [I_amp_1, Q_amp_1, I_amp_0, Q_amp_0, T1]):
        # Mode 3: Generate data with possible relaxation events
        dt = meas_time[1] - meas_time[0]

        for i in range(batch_size):
            rand_val = np.random.random()
            relax_probs = np.exp(-meas_time / T1)
            relax_idx = np.searchsorted(relax_probs[::-1], rand_val)
            relax_idx = (
                len(meas_time) - relax_idx
                if relax_idx < len(meas_time)
                else len(meas_time)
            )

            I_values = np.ones_like(meas_time) * I_amp_1
            Q_values = np.ones_like(meas_time) * Q_amp_1

            if relax_idx < len(meas_time):
                transition_points = max(1, int((relax_time_transition or dt) / dt))
                transition_window = np.linspace(0, 1, transition_points)

                start_idx = relax_idx
                end_idx = min(start_idx + transition_points, len(meas_time))
                window_size = end_idx - start_idx

                I_values[start_idx:end_idx] = (
                    I_amp_1 + (I_amp_0 - I_amp_1) * transition_window[:window_size]
                )
                Q_values[start_idx:end_idx] = (
                    Q_amp_1 + (Q_amp_0 - Q_amp_1) * transition_window[:window_size]
                )

                if end_idx < len(meas_time):
                    I_values[end_idx:] = I_amp_0
                    Q_values[end_idx:] = Q_amp_0

                labels.append({qubit: 2})
            else:
                labels.append({qubit: 1})

            if gauss_noise_amp > 0:
                I_noise = np.random.normal(0, gauss_noise_amp, size=len(meas_time))
                Q_noise = np.random.normal(0, gauss_noise_amp, size=len(meas_time))
                I_values += I_noise
                Q_values += Q_noise

            data[i] = I_values + 1j * Q_values

    else:
        raise ValueError(
            "Invalid parameter combination. Must provide either:\n"
            "1. I_amp_1 and Q_amp_1 only (for no relaxation)\n"
            "2. I_amp_0 and Q_amp_0 only (for relaxed state)\n"
            "3. All amplitude parameters and T1 (for relaxation events)"
        )

    return data, labels


if __name__ == "__main__":
    # Example usage for all three modes with seed
    meas_time = np.linspace(0, 100e-6, 1000)  # 100 microseconds, 1000 points

    # Mode 1: No relaxation (label 1)
    data1, labels1 = generate_relaxation_data(
        batch_size=10,
        meas_time=meas_time,
        I_amp_1=1.0,
        Q_amp_1=0.5,
        gauss_noise_amp=0.1,
        qubit=0,
        seed=42,  # Set seed for reproducibility
    )
    print("Mode 1 (No relaxation):")
    print(f"Data shape: {data1.shape}")
    print(f"First few labels: {labels1[:3]}\n")

    # Mode 2: Relaxed state (label 0)
    data2, labels2 = generate_relaxation_data(
        batch_size=10,
        meas_time=meas_time,
        I_amp_0=0.0,
        Q_amp_0=0.0,
        gauss_noise_amp=0.1,
        qubit=0,
        seed=42,  # Set seed for reproducibility
    )
    print("Mode 2 (Relaxed state):")
    print(f"Data shape: {data2.shape}")
    print(f"First few labels: {labels2[:3]}\n")

    # Mode 3: With relaxation events (labels 1 or 2)
    data3, labels3 = generate_relaxation_data(
        batch_size=10,
        meas_time=meas_time,
        I_amp_1=1.0,
        Q_amp_1=0.5,
        I_amp_0=0.0,
        Q_amp_0=0.0,
        relax_time_transition=1e-6,
        T1=20e-6,
        gauss_noise_amp=0.1,
        qubit=0,
        seed=42,  # Set seed for reproducibility
    )
    print("Mode 3 (With relaxation):")
    print(f"Data shape: {data3.shape}")
    print(f"First few labels: {labels3[:3]}")

    # Demonstrate reproducibility
    data3_repeat, labels3_repeat = generate_relaxation_data(
        batch_size=10,
        meas_time=meas_time,
        I_amp_1=1.0,
        Q_amp_1=0.5,
        I_amp_0=0.0,
        Q_amp_0=0.0,
        relax_time_transition=1e-6,
        T1=20e-6,
        gauss_noise_amp=0.1,
        qubit=0,
        seed=42,  # Same seed as before
    )
    print("\nReproducibility check:")
    print(f"Data is identical: {np.allclose(data3, data3_repeat)}")
    print(f"Labels are identical: {labels3 == labels3_repeat}")
