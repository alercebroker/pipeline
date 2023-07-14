import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

import numpy as np
from modules.coords import ec2gal
import matplotlib.pyplot as plt

ecliptic_lat = np.linspace(0, 0, num=500)
ecliptic_longi = np.linspace(0, 360, num=500)
galact_long, galact_lat = ec2gal(ecliptic_longi, ecliptic_lat)
# sort_idxs = np.argsort(galact_long)
plt.plot(galact_long, galact_lat)
# positions_lat = galact_lat[357]
# print(positions_lat)
# out_pints = np.abs(np.abs(galact_lat)-positions_lat)<0.5
# plt.plot(galact_long[sort_idxs], galact_lat[sort_idxs])
# plt.plot(galact_long[(galact_long<10)], galact_lat[(galact_long<10)])
plt.show()
# print(galact_long<10)
# print(galact_lat>0)
# print(out_pints)
#
# print(galact_lat[np.abs(np.abs(galact_lat)-positions_lat)<0.5])
# print(positions_lat)
# print(np.argwhere((galact_long-np.min(galact_long))==0))
