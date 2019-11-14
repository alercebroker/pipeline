# PAPS

# Installing PAPS

Run:
```
make install
```

# Using PAPS
```
import pandas as pd
import paps

df = pd.read_csv("/home/tronco/Downloads/ZTF18aaajeru_AGN_gband.csv", index_col=0)
mag = df.magpsf_corr
magerr = df.sigmapsf_corr
time = df.index
t1 = 100
t2 = 10

#Other parameters
# dt = 3.0
# mag0 = 19.0
# epsilon = 1.0
ratio,low,high,non_zero, PN_flag = paps.statistics(mag.values,
                                                   magerr.values,
                                                   time.values,
                                                   t1,t2)
```
