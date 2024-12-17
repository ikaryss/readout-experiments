from typing import Iterable
import re
import numpy as np


# function for long array splitting
def divide_signal_into_n(
    signal: np.ndarray,
    meas_time: np.ndarray,
    truncate_start: int = None,
    truncate_end: int = None,
):
    """devides long array into separate parts

    Takes in long array `signal`, separates it into parts wrt `meas_time`.
    Each part is truncated by the indexes from `truncate_start` to `truncate_end`.
    Final output is an array of arrays

    Args:
        signal (np.ndarray): array to cut
        meas_time (np.ndarray): measurement time
        truncate_start (int, optional): starting index. Defaults to None.
        truncate_end (int, optional): ending index. Defaults to None.

    Returns:
        np.ndarray: output 2-d array

    Example:

        >>> import numpy as np

        >>> arr = np.arange(10)
        >>> divide_signal_into_n(arr, 2)

        array([[0., 1., 2., 3., 4.],
               [5., 6., 7., 8., 9.]])

    """
    signal_len = len(signal)
    one_part_len = len(meas_time)
    signals_count = signal_len // one_part_len

    # check if truncate is less then signal length
    if (truncate_start is not None) and (truncate_end is not None):
        assert (
            truncate_start + truncate_end < one_part_len
        ), "truncate should not be greater then one single-shot length"

    n_parts = signal.reshape((signals_count, one_part_len))[
        :, truncate_start:truncate_end
    ]
    return n_parts


def basic_sep(
    in_phase: Iterable,
    quadrature: Iterable,
    meas_time: Iterable,
    truncate_start: int = None,
    truncate_end: int = None,
):

    in_phase = divide_signal_into_n(in_phase, meas_time, truncate_start, truncate_end)

    quadrature = divide_signal_into_n(
        quadrature, meas_time, truncate_start, truncate_end
    )

    return in_phase, quadrature, meas_time[truncate_start:truncate_end]


def arrays_to_complex(in_phase: np.array, quadrature: np.array) -> np.array:
    value = 1j * quadrature
    value += in_phase
    return value.astype(np.complex64)


# def extract_data(path_obj):
#     """
#     Extracts integer IDs and strings before '.pkl' from a pathlib.WindowsPath object.

#     Args:
#         path_obj (pathlib.WindowsPath): The path object to analyze.

#     Returns:
#         tuple: Two lists - integers after 'Q' and strings before '.pkl'.
#               Returns empty lists if not found.
#     """

#     try:
#         # Extract the filename (e.g., 'single_shots_Q1-Q2-Q3_000.pkl')
#         filename = path_obj.name

#         # Use regular expressions to find patterns
#         int_matches = re.findall(r"Q(\d+)", filename)  # Numbers after 'Q'
#         string_matches = re.findall(r"_(\w+)\.pkl", filename)[0]

#         # Convert to integers (and optionally remove duplicates)
#         int_ids = list(set(map(int, int_matches)))

#         return int_ids, list(string_matches)
#     except Exception as e:
#         print(f"Error processing path: {path_obj}. Reason: {e}")
#         return [], []


def extract_data(path_obj):
    # Extract the filename (e.g., 'single_shots_Q1-Q2-Q3_000.pkl')
    # Or 'single_shots_Q1-Q2-Q3_000_Q1.pkl'
    filename = path_obj.name
    last_q_match = re.search(r"Q(\d+)(?=\.pkl)", filename)

    if last_q_match:
        string_match = re.findall(r"_(\d+)_Q\d+\.pkl", filename)[0]
        last_q_match = re.search(r"Q(\d+)(?=\.pkl)", filename)
        last_q_number = int(last_q_match.group(1))
        return [last_q_number], [int(string_match[last_q_number - 1])]

    else:
        string_matches = re.findall(r"_(\w+)\.pkl", filename)[0]
        int_matches = re.findall(r"Q(\d+)", filename)  # Numbers after 'Q'
        int_ids = list(set(map(int, int_matches)))
        return int_ids, list(map(int, string_matches))
