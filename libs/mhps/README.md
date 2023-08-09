[![Build Status](https://travis-ci.com/alercebroker/mhps.svg?branch=master)](https://travis-ci.com/alercebroker/mhps)
[![codecov](https://codecov.io/gh/alercebroker/mhps/branch/master/graph/badge.svg)](https://codecov.io/gh/alercebroker/mhps)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/a5ace81486b44fb18fd332c643976f5e)](https://www.codacy.com/gh/alercebroker/mhps?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=alercebroker/mhps&amp;utm_campaign=Badge_Grade)

# Mexican Hat Power Spectra (MHPS)

Based in work of [`Patricia Arevalo Power Spectra`](https://arxiv.org/abs/1207.5825).

# Installing MHPS

Run:
```
pip install -e .
```

# Using MHPS

```python
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
# Known issues

Some times the algorithm gets an `ZeroDivisionException` this will return all values as numpy NaN (`np.nan`).