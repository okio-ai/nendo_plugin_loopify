import unittest

import numpy as np
from nendo import Nendo, NendoConfig

from nendo_plugin_loopify.utils import (
    LoopMetaData,
    _calc_spectral_sim,
    get_loop_candidates,
    choose_final_loops,
)

nd = Nendo(
    config=NendoConfig(
        library_path="./library",
        log_level="INFO",
        plugins=["nendo_plugin_loopify"],
    )
)


class TestLoopifier(unittest.TestCase):
    def test_run_loopify_plugin(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        loops = nd.plugins.loopify(track=track)

        # default 4 loops + 1 original
        self.assertEqual(len(nd.library.get_tracks()), 5)
        self.assertEqual(len(loops), 4)

    def test_run_process_loopify_plugin(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        loops = track.process("nendo_plugin_loopify")

        # default 4 loops + 1 original
        self.assertEqual(len(nd.library.get_tracks()), 5)
        self.assertEqual(len(loops), 4)


class TestUtils(unittest.TestCase):
    def test_get_loop_candidates_2_beats(self):
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
        beats = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        beats_per_loop = 2

        result = get_loop_candidates(y, beats, beats_per_loop)
        self.assertEqual(len(result), 7)
        self.assertTrue(all(isinstance(loop, LoopMetaData) for loop in result))

    def test_get_loop_candidates_5_beats(self):
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
        beats = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        beats_per_loop = 5

        result = get_loop_candidates(y, beats, beats_per_loop)
        self.assertEqual(len(result), 4)

    def test_get_loop_candidates_20_beats_large_array(self):
        y = np.array(range(50)).astype(np.float32)
        beats = np.array(range(50))
        beats_per_loop = 20

        result = get_loop_candidates(y, beats, beats_per_loop)
        self.assertEqual(len(result), 29)

    def test_get_loop_candidates_0_beats(self):
        y = np.array(range(50)).astype(np.float32)
        beats = np.array(range(50))
        beats_per_loop = 0

        result = get_loop_candidates(y, beats, beats_per_loop)
        self.assertEqual(len(result), 0)

    def test_calc_spectral_sim(self):
        loop = np.array([1, 2, 3, 4, 1]).astype(np.float32)
        window = 1

        result = _calc_spectral_sim(loop, window)
        self.assertEqual(np.round(result, 1), 1.0)

    def test_calc_spectral_sim_large_array(self):
        loop = np.array(range(50)).astype(np.float32)
        window = 1

        result = _calc_spectral_sim(loop, window)
        self.assertEqual(np.round(result), 1.0)

    def test_calc_spectral_sim_large_array_large_window(self):
        loop = np.array(range(50)).astype(np.float32)
        window = 20

        result = _calc_spectral_sim(loop, window)
        self.assertEqual(np.round(result), 1.0)

    def test_choose_final_loops(self):
        loops = [
            LoopMetaData(start_sample=0, end_sample=4, spectral_similarity=0.5),
            LoopMetaData(start_sample=0, end_sample=4, spectral_similarity=0.9),
            LoopMetaData(start_sample=0, end_sample=4, spectral_similarity=0.8),
            LoopMetaData(start_sample=0, end_sample=4, spectral_similarity=1.0),
            LoopMetaData(start_sample=0, end_sample=4, spectral_similarity=0.7),
        ]
        num_loops = 3

        result = choose_final_loops(loops, num_loops)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].spectral_similarity, 1.0)
        self.assertEqual(result[1].spectral_similarity, 0.9)
        self.assertEqual(result[2].spectral_similarity, 0.8)


if __name__ == "__main__":
    unittest.main()
