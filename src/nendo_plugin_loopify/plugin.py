"""Plugin for Nendo that provides music loop generation capabilities."""
from logging import Logger
from typing import Any, List, Optional

from nendo import (
    Nendo,
    NendoConfig,
    NendoGeneratePlugin,
    NendoTrack,
)

from .beatnet import NendoBeatNet
from .utils import choose_final_loops, get_loop_candidates


class Loopifier(NendoGeneratePlugin):
    """A plugin for Nendo that provides music loop generation capabilities.

    Attributes:
        nendo_instance (Nendo): The instance of Nendo using this plugin.
        config (NendoConfig): The configuration of the Nendo instance.
        logger (Logger): The logger to use for reporting.


    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_loopify"]))
        track = nd.library.add_track(file_path='/path/to/track.mp3')

        generated_loops = nd.plugins.loopify(
            track=track,
            num_beats=4,
            beats_per_loop=8
        )
        generated_loops[0].loop()
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    logger: Logger = None
    beatnet: NendoBeatNet = None

    def __init__(self, **data: Any):
        """Initialize the loopifier plugin, setting up BeatNet."""
        super().__init__(**data)
        self.beatnet = NendoBeatNet()

    @NendoGeneratePlugin.run_track
    def loopify_track(
            self,
            track: NendoTrack,
            n_loops: Optional[int] = 4,
            beats_per_loop: Optional[int] = 8,
    ) -> List[NendoTrack]:
        """Run the BeatNet loopifier on the given track.

        Args:
            track (NendoTrack): The track to loopify.
            n_loops (int, optional): The number of loops to generate.
            beats_per_loop (int, optional): The number of beats per loop.

        Returns:
            List[NendoTrack]: The generated loops.
        """
        loops: List[NendoTrack] = []

        y, sr = track.signal, track.sr
        self.logger.debug(f"Loaded track with shape {y.shape} and sample rate {sr}.")

        # only process mono signal
        beatmap = self.beatnet.process(y[0])
        beats = (beatmap[:, 0] * sr).astype(int)

        if len(beats) == 0:
            self.logger.warning("No beats detected in track.")

        loop_candidates = get_loop_candidates(y, beats, beats_per_loop)

        if len(loop_candidates) == 0:
            self.logger.warning("No loops found in track.")

        self.logger.debug(f"Found {len(loop_candidates)} loop candidates.")

        final_loops = choose_final_loops(loop_candidates, n_loops)
        self.logger.debug(f"Found final loops:\n{final_loops}")

        for i, loop in enumerate(final_loops):
            loop_buffer = (
                y[:, loop.start_sample: loop.end_sample]
                if len(y.shape) > 1
                else y[loop.start_sample: loop.end_sample]
            )
            self.logger.debug(f"Loop buffer shape: {loop_buffer.shape}")

            if "original_filename" not in track.resource.meta:
                track_title = f"Loop {i + 1}"
            else:
                track_title = f"{track.resource.meta['original_filename']} - Loop {i + 1}"
            loops.append(
                self.nendo_instance.library.add_related_track_from_signal(
                    signal=loop_buffer,
                    sr=sr,
                    track_type="loop",
                    relationship_type="loop",
                    track_meta={"title": track_title},
                    related_track_id=track.id,
                ),
            )

        return loops
