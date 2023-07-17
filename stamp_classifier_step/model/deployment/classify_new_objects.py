# import cProfile
import time
import pandas as pd

from stamp_clf import StampClassifier
from avro_to_pickle import LastDayAvros
from IPython.core.debugger import set_trace

last_day_avros = "/home/ireyes/alerce_stamps/stamps_20190624"
converter = LastDayAvros(last_day_avros, "ZTF19abamyzw", batch_size=10000)

clf = StampClassifier()

dfs = []
# profiler = cProfile.Profile()
# profiler.enable()
t = time.time()
for i, avros_batch in enumerate(converter.get_avros_batch()):
    avros_batch = avros_batch.loc[~avros_batch.index.duplicated(keep="first")]
    pred = clf.execute(avros_batch)
    pred_and_stamps = pred.join(avros_batch, how="inner")
    dfs.append(pred_and_stamps)
    at = time.time()
    print("batch", i, at - t, "seconds")
    t = at
# profiler.disable()
# profiler.dump_stats('last_day.prof')

df = pd.concat(dfs)
df = df.loc[~df.index.duplicated(keep="first")]
df.to_pickle("one_stamp_%s.pkl" % (last_day_avros.split("_")[-1]))
