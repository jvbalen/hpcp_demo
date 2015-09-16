HPCP
====

The enclosed Python script can be used to extract chroma features from an audio file. It accepts wav-files, and outputs a json-formatted string containing the chroma frames. Chroma is computed following the HPCP algorithm, which is distinct in that it is based on spectral peaks only, and because harmonic frequencies are summed together, among other contributions.

## dependencies

HPCP.py relies on three common libraries: Numpy, Scipy (scipy.io, scipy.signal and scipy.sparse) and JSON for output. These can be installed in several ways, see http://www.scipy.org/install.html. All other code is new and my own.

## usage

Command line:
Run from command line with a filename (wav) as an argument, e.g.:
```
python HPCP.py 'bach.wav'
```

Python interpreter:
```
import HCPC
chroma = HPCP.hpcp('bach.wav', norm_frames=True)
print chroma
```
Consult `HPCP.hpcp?` for more parameters.


Copyright (c) 2015 jvbalen / @jvanbalen under the MIT License
Github: https://github.com/jvbalen/hpcp_notebook