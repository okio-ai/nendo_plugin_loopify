# Advanced Usage

### How we do loop search

For finding loops, we first extract beats and downbeats from the track.
After that we step through all beats with a given `beats_per_loop` and generate all possible candidates.

For each candidate we then compare the spectral similarity of the start and end of the loop.
Finally, we return the top `num_beats` candidates.

Play around with the `beats_per_loop` parameter to get different results. 
You can go up to a few hundred beats per loop to get really long loopable sections and interesting results.

!!! tip
    We recommend using `beats_per_loop` between 4 and 32. To get results that feel like loops the most.
