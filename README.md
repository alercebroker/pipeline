# Mexican Hat Power Spectra (MHPS)
[![Build Status](https://travis-ci.com/alercebroker/mhps.svg?branch=master)](https://travis-ci.com/alercebroker/mhps)

Based in work of [`Patricia Arevalo Power Spectra`](https://arxiv.org/abs/1207.5825).

# Installing MHPS

Run:
```
pip install -e .
```

# Using MHPS

```
import pandas as pd
import mhps

df = pd.read_csv(".../ZTF18aaajeru.csv", index_col=0)
mag = df.magpsf_corr
magerr = df.sigmapsf_corr
time = df.index
t1 = 100
t2 = 10

#Other parameters
# dt = 3.0
# mag0 = 19.0
# epsilon = 1.0
ratio,low,high,non_zero, PN_flag = mhps.statistics(mag.values,
                                                   magerr.values,
                                                   time.values,
                                                   t1,t2)
```
