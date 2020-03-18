import pandas as pd
import mhps

df = pd.read_csv("ZTF18aaajeru.csv")

# Select only fid == 1 (g-band)
df = df[df.fid == 1]

mag = df.magpsf_corr
magerr = df.sigmapsf_corr
time = df.mjd
t1 = 100
t2 = 10

# Other parameters
# dt = 3.0
# mag0 = 19.0
# epsilon = 1.0
ratio,low,high,non_zero, PN_flag = mhps.statistics(mag.values,
                                                   magerr.values,
                                                   time.values,
                                                   t1,
                                                   t2)
