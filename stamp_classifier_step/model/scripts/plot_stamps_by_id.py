from raw_data_manage.alerce_api import AlerceApi, AlerceStamps
from modules.data_loaders.frame_to_input import get_image_from_bytes_stamp
import matplotlib.pyplot as plt
import pandas as pd
import os


alert_list = ["ZTF19abmolyr"]
alerce = AlerceStamps(".")
alerce.download_avros(object_ids=alert_list, file_name="alert_frame_list.pkl")
features_to_add = []
frame = alerce.create_frame_to_merge(
    frame_class="SN", alerts_list="alert_frame_list.pkl", features=features_to_add
)
print(frame)

stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]
for key in stamp_keys:
    stamp = get_image_from_bytes_stamp(frame.iloc[0][key])
    plt.imshow(stamp, cmap="inferno")
    plt.axis("off")
    plt.savefig(key + ".pdf", bbox_inches="tight")

os.system("rm alert_frame_list alert_ZTF19abmolyr_946476632015015009.avro")
