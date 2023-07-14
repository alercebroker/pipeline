import os
import pandas as pd
import fastavro
import pickle


class AvroConverter(object):
    def __init__(self, stamps_dir, batch_size=10000):
        self.stamps_dir = stamps_dir
        self.batch_size = batch_size
        self.stamp_keys = ["cutoutScience", "cutoutTemplate", "cutoutDifference"]

    def get_avros_batch(self):
        oids = []
        data = []
        i = 0
        for root, dirs, files in os.walk(self.stamps_dir):
            if len(files) > 0:
                numbers = [f.split(".")[0] for f in files]
                first_stamp = str(min(numbers)) + ".avro"
                filename = os.path.join(root, first_stamp)
                oid = "".join(filename[len(self.stamps_dir) :].split("/")[:-1])

                oids.append(oid)
                data.append(self.read_avro(filename))
                i += 1
                if i % self.batch_size == 0:
                    yield pd.DataFrame(index=oids, columns=self.stamp_keys, data=data)
                    oids = []
                    data = []

        yield pd.DataFrame(index=oids, columns=self.stamp_keys, data=data)

    def read_avro(self, avro_path):
        with open(avro_path, "rb") as f:
            freader = fastavro.reader(f)
            schema = freader.schema
            for i, packet in enumerate(freader):
                continue
            data = [packet[key]["stampData"] for key in self.stamp_keys]
        return data

    def merge_dataframes(self):
        self.auxiliar_frame = pd.concat(
            [self.auxiliar_frame, self.images_frame], axis=1
        )
        print(self.auxiliar_frame)


class LastDayAvros(AvroConverter):
    def __init__(self, stamps_dir, last_oid, batch_size=1000):
        super(LastDayAvros, self).__init__(stamps_dir, batch_size=batch_size)
        self.last_oid = last_oid

    def get_avros_batch(self):
        oids = []
        data = []
        i = 0
        for root, dirs, files in os.walk(self.stamps_dir):
            if len(files) > 0:
                for file in files:
                    filename = os.path.join(root, file)
                    try:
                        ims, oid = self.read_avro_ims_oid(filename)
                    except:
                        print(f"File {filename} has failed (read_avro_ims_oid)")
                        continue
                    if oid > self.last_oid:
                        oids.append(oid)
                        data.append(ims)
                        i += 1
                        if i % self.batch_size == 0:
                            yield pd.DataFrame(
                                index=oids, columns=self.stamp_keys, data=data
                            )
                            oids = []
                            data = []

        yield pd.DataFrame(index=oids, columns=self.stamp_keys, data=data)

    def read_avro_ims_oid(self, avro_path):
        with open(avro_path, "rb") as f:
            freader = fastavro.reader(f)
            for i, packet in enumerate(freader):
                continue
            ims = [packet[key]["stampData"] for key in self.stamp_keys]
            oid = packet["objectId"]
        return ims, oid


if __name__ == "__main__":
    avro_file = "/home/ireyes/alerce_stamps/ZTF17/a/a/a/a/a/k/x/528426604115010005.avro"
    converter = AvroConverter()
    print(converter.read_avro(avro_file).keys())
