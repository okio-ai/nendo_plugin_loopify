"""NendoBeatNet implementation."""
from typing import List, Optional, Union

from madmom.audio import (
    SignalProcessor,
    FramedSignalProcessor,
    ShortTimeFourierTransformProcessor,
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom.features import DBNDownBeatTrackingProcessor
from madmom.processors import ParallelProcessor, SequentialProcessor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class NendoBeatNet:
    """The main BeatNet class adapted from the original implementation at https://github.com/mjhydri/BeatNet.

    Returns:
        A vector including beat times and downbeat identifier columns, respectively with the following shape: numpy_array(num_beats, 2).
    """

    def __init__(self):
        """Initialize the BeatNet class."""
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LogSpec(
            sample_rate=self.log_spec_sample_rate,
            win_length=self.log_spec_win_length,
            hop_size=self.log_spec_hop_length,
            n_bands=[24],
        )
        self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        script_dir = os.path.dirname(__file__)
        self.model = BeatDownBeatActivation(
            272,
            150,
            2,
            self.device,
        )  # Beat Downbeat Activation detector

        self.model.load_state_dict(
            torch.load(os.path.join(script_dir, "models/model.pt")),
            strict=False,
        )
        self.model.eval()

    def process(self, audio_path: Optional[Union[str, np.ndarray]] = None) -> np.ndarray:
        """Process the given audio file or object and return the beat times and downbeat identifier columns."""
        if isinstance(audio_path, str) or audio_path.all() is not None:
            preds = self._activation_extractor_online(audio_path)
            return self.estimator(preds)
        raise RuntimeError(
            "An audio object or file directory is required for the offline usage!",
        )

    def _activation_extractor_online(self, audio) -> torch.Tensor:
        with torch.no_grad():
            # we only process every second beat, same as in the original implementation
            feats = self.proc.process_audio(audio[::2]).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0).to(self.device)
            preds = self.model(feats)[0]
            preds = self.model.final_pred(preds)
            preds = preds.cpu().detach().numpy()
            return np.transpose(preds[:2, :])


class LogSpec:
    """LogSpec class adapted from the original implementation."""
    def __init__(
        self,
        num_channels: int = 1,
        sample_rate: int = 22050,
        win_length: int = 2048,
        hop_size: int = 512,
        n_bands: List[int] = [12],
    ):
        """Initialize the LogSpec class."""
        sig = SignalProcessor(
            num_channels=num_channels,
            win_length=win_length,
            sample_rate=sample_rate,
        )
        self.sample_rate = sample_rate
        self.hop_length = hop_size
        self.num_channels = num_channels
        multi = ParallelProcessor([])
        frame_sizes = [win_length]
        num_bands = n_bands
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size)
            stft = ShortTimeFourierTransformProcessor()  # caching FFT window
            filt = FilteredSpectrogramProcessor(
                num_bands=num_bands,
                fmin=30,
                fmax=17000,
                norm_filters=True,
            )
            spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
            diff = SpectrogramDifferenceProcessor(
                diff_ratio=0.5,
                positive_diffs=True,
                stack_diffs=np.hstack,
            )
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # stack the features and processes everything sequentially
        self.pipe = SequentialProcessor((sig, multi, np.hstack))

    def process_audio(self, audio):
        """Process the given audio file or object and return the log spectrogram."""
        feats = self.pipe(audio)
        return feats.T


def num_flat_features(x: torch.Tensor) -> int:
    """Return the number of features in a tensor."""
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class BeatDownBeatActivation(nn.Module):
    """Beat Downbeat Activation model."""
    def __init__(
        self, dim_in: int, num_cells: int, num_layers: int, device: torch.device,
    ):
        """Initialize the BeatDownBeatActivation model."""
        super(BeatDownBeatActivation, self).__init__()

        self.dim_in = dim_in
        self.dim_hd = num_cells
        self.num_layers = num_layers
        self.device = device
        self.conv_out = 150
        self.kernelsize = 10
        self.conv1 = nn.Conv1d(1, 2, self.kernelsize)
        self.linear0 = nn.Linear(
            2 * int((self.dim_in - self.kernelsize + 1) / 2),
            self.conv_out,
        )  # divide to 2 is for max pooling filter
        self.lstm = nn.LSTM(
            input_size=self.conv_out,  # self.dim_in
            hidden_size=self.dim_hd,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.linear = nn.Linear(in_features=self.dim_hd, out_features=3)

        self.softmax = nn.Softmax(dim=0)
        # Initialize the hidden state and cell state
        self.hidden = torch.zeros(2, 1, self.dim_hd).to(device)
        self.cell = torch.zeros(2, 1, self.dim_hd).to(device)

        self.to(device)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = data
        x = torch.reshape(x, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, num_flat_features(x))
        x = self.linear0(x)
        x = torch.reshape(x, (np.shape(data)[0], np.shape(data)[1], self.conv_out))
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        # x = self.lstm(x)[0]
        out = self.linear(x)
        return out.transpose(1, 2)

    def final_pred(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the final prediction from the model."""
        return self.softmax(input_tensor)
