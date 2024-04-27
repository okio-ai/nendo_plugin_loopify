"""Utility functions for the loop generation process."""
import dataclasses
import math
from typing import List
from enum import Enum
from nendo import NendoPluginRuntimeError

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

class QualityScore(Enum):
    """Quality score used for loop selection."""
    SPECTRAL = "spectral"
    SPECTRAL_DISTANCE = "spectral_distance"

def normalize_scores(scores: List[float]) -> List[float]:
    """Normalizes a list of scores to a 0-1 range."""
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5 for _ in scores]  # Avoid division by zero if all scores are the same
    return [(score - min_score) / (max_score - min_score) for score in scores]

def choose_final_loops(
    loop_candidates: List[LoopMetaData],
    num_loops: int,
    score: QualityScore = QualityScore.SPECTRAL_DISTANCE,
) -> List[LoopMetaData]:
    """Chooses the final loops to be used.

    Right now only ranks the loops by spectral similarity and chooses the top num_loops.
    In the future this function will be expanded by heuristics to choose the best loops.

    Args:
        loop_candidates (list[LoopMetaData]): The candidate loops to choose from.
        num_loops (int): The number of loops to choose.
        score (QualityScore): The quality score used when selecting the loops.
            Defaults to the combination of normalized spectral similarity
            and distance score.

    Returns:
        list[LoopMetaData]: The chosen loops.
    """
    if score == QualityScore.SPECTRAL:
        print("DOING SPECTRAL")
        loop_candidates.sort(key=lambda x: x.spectral_similarity, reverse=True)
        return loop_candidates[:num_loops]
    elif score == QualityScore.SPECTRAL_DISTANCE:
        # Sort candidates by spectral similarity for initial selection potential
        loop_candidates.sort(key=lambda x: x.spectral_similarity, reverse=True)

        # Initialize the selection with the loop having the highest similarity
        selected_loops = [loop_candidates[0]]

        while len(selected_loops) < num_loops and len(selected_loops) < len(loop_candidates):
            best_score = float('-inf')
            best_candidate = None

            diversity_scores = [
                min(
                    np.abs(candidate.start_sample - selected.start_sample) + np.abs(candidate.end_sample - selected.end_sample)
                    for selected in selected_loops
                ) for candidate in loop_candidates if candidate not in selected_loops
            ]

            # Normalize both spectral similarity and diversity scores
            if diversity_scores:
                normalized_diversity_scores = normalize_scores(diversity_scores)
                normalized_similarity_scores = normalize_scores([c.spectral_similarity for c in loop_candidates if c not in selected_loops])

                # Iterate over candidates not already selected
                for i, candidate in enumerate([c for c in loop_candidates if c not in selected_loops]):
                    # Combine normalized scores with equal weighting
                    combined_score = 0.5 * normalized_similarity_scores[i] + 0.5 * normalized_diversity_scores[i]

                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate

            if best_candidate:
                selected_loops.append(best_candidate)

        return selected_loops
    else:
        raise NendoPluginRuntimeError("Loop selection score unknown")


def _calc_spectral_sim(spec: torch.Tensor, window: int = 1) -> float:
    """Calculates a given loops quality based on Spectral continuity between the first and last window of the loop.

    Args:
        spec (torch.Tensor): The spectrogram of the loop.
        window (int, optional): The window size for the spectral similarity calculation. Defaults to 1.

    Returns:
        float: The spectral similarity of the loop.
    """
    frame_start = spec[:, :window].flatten()
    frame_end = spec[:, -window:].flatten()
    frame_start_norm = frame_start / torch.linalg.norm(frame_start)
    frame_end_norm = frame_end / torch.linalg.norm(frame_end)
    spectral_sim = torch.dot(frame_start_norm, frame_end_norm)
    return spectral_sim.item()


def _calc_spectrogram(loop: np.ndarray) -> torch.Tensor:
    """Calculates the spectrogram of a given loop.

    Args:
        loop (numpy.ndarray): The loop to calculate the spectrogram for.

    Returns:
        torch.Tensor: The spectrogram of the loop.
    """
    return torch.abs(
        torch.stft(
            torch.from_numpy(loop),
            n_fft=2048,
            return_complex=True,
            pad_mode="constant",
        ),
    )


def _mean_rms(S: torch.Tensor) -> torch.Tensor:
    """Calculates the mean RMS of a given spectrogram."""
    rms = torch.sqrt(torch.mean(S**2, dim=0))
    return torch.mean(rms)


def get_loop_candidates(
    y: np.ndarray,
    beats: np.ndarray,
    beats_per_loop: int,
    dynamic_threshold_factor: float = 0.5,
    silence_threshold: float = 0.005,
) -> List[LoopMetaData]:
    """Generate potential loops from a given track.

    Args:
        y (numpy.ndarray): The track data.
        beats (numpy.ndarray): The beatmap of the track.
        beats_per_loop (int): The number of beats per loop.
        dynamic_threshold_factor (float, optional): The factor to multiply the dynamic threshold by. Defaults to 0.5.
        silence_threshold (float, optional): The threshold to consider a loop silent. Defaults to 0.005.

    Returns:
        list[LoopMetaData]: The potential loops generated.
    """
    beat_intervals = np.diff(beats)
    loop_candidates = []
    spec_total = _calc_spectrogram(y)
    rms_total = _mean_rms(spec_total)
    dynamic_threshold = rms_total * dynamic_threshold_factor
    for i in range(len(beat_intervals) - beats_per_loop):
        start = beats[i]
        end = beats[i + beats_per_loop]
        loop = y[:, start:end] if len(y.shape) > 1 else y[start:end]
        spec_loop = _calc_spectrogram(loop)
        spectral_sim = _calc_spectral_sim(spec_loop)
        rms_loop = _mean_rms(spec_loop)
        if rms_loop > max(dynamic_threshold, silence_threshold):
            loop_candidates.append(
                LoopMetaData(
                    start_sample=start,
                    end_sample=end,
                    spectral_similarity=spectral_sim,
                ),
            )
    return loop_candidates
