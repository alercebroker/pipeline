# import cProfile
import time
import pandas as pd
from stamp_clf import StampClassifier
from avro_to_pickle import AvroConverter


avro_dir = "/home/ireyes/alerce_stamps/stamps"
converter = AvroConverter(avro_dir)

clf = StampClassifier()

dfs = []
# profiler = cProfile.Profile()
# profiler.enable()
t = time.time()
for i, avros_batch in enumerate(converter.get_avros_batch()):
    pred = clf.execute(avros_batch)
    dfs.append(pred)
    at = time.time()
    print("batch", i, at - t, "seconds")
    t = at
# profiler.disable()
# profiler.dump_stats('process_historic_data.prof')

df = pd.concat(dfs)
df.to_pickle("one_stamp_predicions.pkl")
