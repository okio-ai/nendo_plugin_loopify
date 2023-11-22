# Nendo Plugin Loopify

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

---

![Documentation](https://img.shields.io/website/https/nendo.ai)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai)](https://twitter.com/okio_ai) [![](https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat)](https://discord.gg/XpkUsjwXTp)


Automatic audio loop extraction.


## Features

- Automatically find loopable sections in a `NendoTrack` 
- Automate your sample digging process

## Requirements

Due to `madmom` versions < 0.17 errors with python 3.10, we require the latest version of the  package from git, where this is fixed. See also [this related issue](https://github.com/CPJKU/madmom/issues/502).

Run:

`pip install git+https://github.com/CPJKU/madmom.git@0551aa8`

## Installation

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-loopify`

## Usage

Take a look at a basic usage example below.
For more detailed information, please refer to the [documentation](https://okio.ai/docs/plugins).

For more advanced examples, check out the examples folder.
or try it in colab:

<a target="_blank" href="https://colab.research.google.com/drive/1OD38SedBRHhOYpwGpzni2SJv-Osxu__i?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_loopify"]))
track = nd.library.add_track(file_path='/path/to/track.mp3')

generated_loops = nd.plugins.loopify(
    track=track,
    n_loops=4,
    beats_per_loop=8
)
generated_loops[0].loop()
```

## Contributing

Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)

## License 

Nendo: MIT License

Madmom: BSD License

Pretrained models: The weights are released under the CC-BY-NC 4.0 license