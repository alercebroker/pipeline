from turbofats import FeatureSpace
import numpy as np
import pandas as pd
import time


lc_data = pd.read_pickle('ZTF18aaiopei_detections.pkl')
lc_g = lc_data[lc_data.fid == 1]
lc_r = lc_data[lc_data.fid == 2]
lc_g_np = lc_g[['magpsf_corr', 'mjd', 'sigmapsf_corr']].values.T

feature_space = FeatureSpace(['IAR_phi'])

times = []
for i in range(10000):
    t0 = time.time()
    features = feature_space.calculate_features(lc_g)
    tf = time.time()
    times.append(tf - t0)

times = np.array(times)
print(f'Time elapsed {times.mean():.4f} ({times.std():.4f})[s]')
