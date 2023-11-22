"""Utility functions for the loop generation process."""
import dataclasses
import math
from typing import List

import numpy as np
import torch


@dataclasses.dataclass
class LoopMetaData:
    """Dataclass for storing loop metadata.

    Attributes:
        start_sample (int): Starting sample point of the loop.
        end_sample (int): Ending sample point of the loop.
        spectral_similarity (float): Spectral similarity of the loop.
    """

    start_sample: int
    end_sample: int
    spectral_similarity: float


def choose_final_loops(
    loop_candidates: List[LoopMetaData], num_loops: int,
) -> List[LoopMetaData]:
    """Chooses the final loops to be used.

    Right now only ranks the loops by spectral similarity and chooses the top num_loops.
    In the future this function will be expanded by heuristics to choose the best loops.

    Args:
        loop_candidates (list[LoopMetaData]): The candidate loops to choose from.
        num_loops (int): The number of loops to choose.

    Returns:
        list[LoopMetaData]: The chosen loops.
    """
    loop_candidates.sort(key=lambda x: x.spectral_similarity, reverse=True)
    return loop_candidates[:num_loops]


def _calc_spectral_sim(loop: np.ndarray, window: int = 1) -> float:
    """Calculates a given loops quality based on Spectral continuity between the first and last window of the loop.

    Args:
        loop (numpy.ndarray): The loop to calculate the spectral similarity for.
        window (int, optional): The window size for the spectral similarity calculation. Defaults to 1.

    Returns:
        float: The spectral similarity of the loop.
    """
    spec = torch.abs(
        torch.stft(
            torch.from_numpy(loop), n_fft=2048, return_complex=True, pad_mode="constant",
        ),
    )
    frame_start = spec[:, :window].flatten()
    frame_end = spec[:, -window:].flatten()
    frame_start_norm = frame_start / torch.linalg.norm(frame_start)
    frame_end_norm = frame_end / torch.linalg.norm(frame_end)
    spectral_sim = torch.dot(frame_start_norm, frame_end_norm)
    return spectral_sim.item()


def get_loop_candidates(
    y: np.ndarray,
    beats: np.ndarray,
    beats_per_loop: int,
) -> List[LoopMetaData]:
    """Generate potential loops from a given track.

    Args:
        y (numpy.ndarray): The track data.
        beats (numpy.ndarray): The beatmap of the track.
        beats_per_loop (int): The number of beats per loop.

    Returns:
        list[LoopMetaData]: The potential loops generated.
    """
    beat_intervals = np.diff(beats)
    loop_candidates = []
    for i in range(len(beat_intervals) - beats_per_loop):
        start = beats[i]
        end = beats[i + beats_per_loop]
        loop = y[:, start:end] if len(y.shape) > 1 else y[start:end]
        spectral_sim = _calc_spectral_sim(loop)
        if not math.isnan(spectral_sim):
            loop_candidates.append(
                LoopMetaData(
                    start_sample=start,
                    end_sample=end,
                    spectral_similarity=spectral_sim,
                ),
            )
    return loop_candidates
