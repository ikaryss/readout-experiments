"""Curriculum learning strategy for IQ signal denoising."""

from dataclasses import dataclass
from typing import Tuple, List, Iterable
import numpy as np
from denoiser.config import *
from data_generator import IQParameters, QuantumStateGenerator, RelaxationParameters


@dataclass
class CurriculumStage:
    """Parameters for each curriculum learning stage."""

    in_phase_range: float | Iterable
    quadrature_range: float | Iterable

    noise_amp: float | Iterable
    batch_ground: int
    batch_excited: int
    batch_relaxation: int
    epochs: int

    def __str__(self):
        return (
            f"Stage(noise={self.noise_amp}, "
            f"signal={self.in_phase_range}, "
            f"ground={self.batch_ground}, "
            f"excited={self.batch_excited}, "
            f"relax={self.batch_relaxation})"
        )


def get_curriculum_stages() -> List[CurriculumStage]:
    """Define curriculum learning stages."""
    return [
        CurriculumStage(
            in_phase_range=in_phase_range,
            quadrature_range=quadrature_range,
            noise_amp=noise_amp,
            batch_ground=batch_ground,
            batch_excited=batch_excited,
            batch_relaxation=batch_relaxation,
            epochs=epochs,
        )
        for in_phase_range, quadrature_range, noise_amp, batch_ground, batch_excited, batch_relaxation, epochs in zip(
            CurriculumStages.IN_PHASE_RANGES,
            CurriculumStages.QUADRATURE_RANGES,
            CurriculumStages.STAGES_NOISE_AMP,
            CurriculumStages.BATCHES_GROUND,
            CurriculumStages.BATCHES_EXCITED,
            CurriculumStages.BATCHES_RELAX,
            CurriculumStages.EPOCHS,
        )
    ]


def generate_curriculum_data(stage: CurriculumStage) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data for specific curriculum stage."""
    # Use your existing parameters
    excited = IQParameters(I_amp=stage.in_phase_range, Q_amp=stage.quadrature_range)
    ground = IQParameters(I_amp=stage.in_phase_range, Q_amp=stage.quadrature_range)
    relaxation = RelaxationParameters(
        T1=T1_TIME, relax_time_transition=RELAX_TRANSITION_TIME
    )

    # Initialize generator with stage-specific noise
    generator = QuantumStateGenerator(
        meas_time=MEAS_TIME,
        excited_params=excited,
        ground_params=ground,
        relaxation_params=relaxation,
        gauss_noise_amp=stage.noise_amp,
        qubit=QUBIT,
    )

    # Generate data according to stage parameters
    noisy_data = []
    clean_data = []

    if stage.batch_ground > 0:
        ground_noisy, ground_labels_noisy = generator.generate_ground_state(
            batch_size=stage.batch_ground, seed=DATA_SEED
        )
        noisy_data.append(ground_noisy)

    if stage.batch_excited > 0:
        excited_noisy, excited_labels_noisy = generator.generate_excited_state(
            batch_size=stage.batch_excited, seed=DATA_SEED
        )
        noisy_data.append(excited_noisy)

    if stage.batch_relaxation > 0:
        relax_noisy, relax_labels_noisy = generator.generate_relaxation_event(
            batch_size=stage.batch_relaxation, uniform_sampling=True, seed=DATA_SEED
        )
        noisy_data.append(relax_noisy)

    # Generate clean data (low noise)
    generator.gauss_noise_amp = CLEAN_AMP  # Low noise for clean data

    if stage.batch_ground > 0:
        ground_clean, ground_labels_clean = generator.generate_ground_state(
            batch_size=stage.batch_ground, seed=DATA_SEED
        )
        assert ground_labels_noisy == ground_labels_clean
        clean_data.append(ground_clean)

    if stage.batch_excited > 0:
        excited_clean, excited_labels_clean = generator.generate_excited_state(
            batch_size=stage.batch_excited, seed=DATA_SEED
        )
        assert excited_labels_noisy == excited_labels_clean
        clean_data.append(excited_clean)

    if stage.batch_relaxation > 0:
        relax_clean, relax_labels_clean = generator.generate_relaxation_event(
            batch_size=stage.batch_relaxation, uniform_sampling=True, seed=DATA_SEED
        )
        assert relax_labels_noisy == relax_labels_clean
        clean_data.append(relax_clean)

    # Combine and shuffle data
    noisy_data = np.concatenate(noisy_data)
    clean_data = np.concatenate(clean_data)

    shuffle_idx = np.random.permutation(len(noisy_data))
    return noisy_data[shuffle_idx], clean_data[shuffle_idx]


if __name__ == "__main__":
    # Test curriculum stages
    stages = get_curriculum_stages()
    for i, stage in enumerate(stages, 1):
        print(f"\nStage {i}:")
        print(stage)