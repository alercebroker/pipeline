import pandas as pd
import logging
import glob
import tqdm
import sys
import os

sys.path.append("../")
from dotenv import load_dotenv

from alerce_classifiers.base.factories import input_dto_factory
from alerce_classifiers.stamp_full.mapper import StampFullMapper
from alerce_classifiers.stamp_full.model import StampClassifierFull

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

load_dotenv()

def run(path_stamps, 
        #path_partition,
        subset,
        batch_size,
        fold=0):
    
    # Load data
    all_stamps = pd.read_pickle(path_stamps)
    oids = all_stamps.index.values
    logging.info(f"Data loading complete. Number of oids: {len(oids)}.")

    oid_batches = [
        oids[i:i + batch_size]
        for i in range(0, len(oids), batch_size)
    ]

    # Make results directory
    dir_results = '_'.join(os.getenv("TEST_STAMP_FULL_MODEL_PATH").split('/')[-3:-1])
    os.makedirs(f'./results/{dir_results}', exist_ok=True)
    logging.info(f"Results directory created at ./results/{dir_results}")

    # Inference process
    logging.info("Starting the inference process...")
    OutputDTO_dict = {subset: []}
    for oid_batch in tqdm.tqdm(oid_batches):
        stamps = all_stamps.loc[oid_batch]

        # InputDTO with defined batch size 
        input_dto = input_dto_factory(
            detections=pd.DataFrame(), 
            non_detections=pd.DataFrame(), 
            features=pd.DataFrame(), 
            xmatch=pd.DataFrame(), 
            stamps=stamps
        )

        # Model
        model = StampClassifierFull(
            model_path=os.getenv("TEST_STAMP_FULL_MODEL_PATH"),
            #model_path='model.keras',
            mapper=StampFullMapper(),
        )

        # Predictions
        predictions = model.predict(input_dto)
        OutputDTO_dict[subset].append(predictions.probabilities)

    # Save results
    df_probabilities = pd.concat(OutputDTO_dict[subset])
    output_path = f'./results/{dir_results}/predictions_{subset}.parquet'
    df_probabilities.to_parquet(output_path)

    logging.info("Inference process completed successfully.")
    logging.info(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    ROOT = "../../../data_acquisition/data"

    path_stamps = "{}/test_first_stamps_dataset.pkl".format(ROOT)
    #path_partition = '{}/preprocessed/partitions/241209/partitions.parquet'.format(ROOT)

    batch_size = 50

    ## NO ESTA FUNCIONANDO AHORA MISMO
    subset = 'test' # ['training', 'validation', 'test']

    run(path_stamps, 
        #path_partition,
        subset,
        batch_size)