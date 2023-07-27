import os
import logging
import numpy as np
import shutil


def concat_avros(input_directory, output_directory):
    names = os.listdir(input_directory)
    logger = logging.getLogger("concat_avros")
    logger.info(f"{len(names)} avro files on {input_directory}")
    chunk_size = 2200
    partitions = int(np.ceil(len(names) / chunk_size))
    print(os.getcwd())
    os.mkdir(output_directory)
    os.chdir(input_directory)
    chunks = np.array_split(names, partitions)

    count = 0
    for chunk in chunks:
        logger.info(f"Generating partition_{count}")
        files = " ".join(chunk.tolist())
        output_path = os.path.join(output_directory, "partition_%d.avro" % (count))
        logger.info(f"Saving concatenated avro in {output_path}")
        command = "java -jar ../lib/avro-tools-1.8.2.jar concat %s %s" % (
            files,
            output_path,
        )
        os.system(command)
        logger.info(f"Partition {count} generated")
        count = count + 1
    os.chdir("..")
    print(os.getcwd())
