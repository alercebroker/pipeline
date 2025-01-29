import pandas as pd
import logging
import tqdm
import sys
import os

sys.path.append("../")
from utils import *
from dotenv import load_dotenv

from alerce_classifiers.base.factories import input_dto_factory
from alerce_classifiers.squidward.mapper import SquidwardMapper
from alerce_classifiers.squidward.model import SquidwardFeaturesClassifier

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

load_dotenv()

def run(dir_astro_lightcurves, dir_astro_features, path_partition, subset, batch_size, 
         use_multiprocessing, num_cores, fold=0):
    
    # Load data
    path_chunks = glob.glob(f"{dir_astro_lightcurves}/astro_objects_batch_*")
    args = [(path_chunk, dir_astro_features) for path_chunk in path_chunks]
    all_detections, all_features = load_astro_objects_as_InputDTO(
        args, use_multiprocessing, num_cores
    )
    logging.info(f"Data loading complete. Number of oids: {len(all_features)}.")

    # Get subset from partitions
    all_detections, all_features, oid_batches = get_subset_and_batches(
        all_detections, all_features, path_partition, subset, fold, batch_size
        )
    logging.info(f"Number of oids in {subset}: {len(all_features)}.")

    # Make results directory
    dir_results = '_'.join(os.getenv("TEST_SQUIDWARD_MODEL_PATH").split('/')[-3:-1])
    os.makedirs(f'./results/{dir_results}', exist_ok=True)
    logging.info(f"Results directory created at ./results/{dir_results}")

    # Inference process
    logging.info("Starting the inference process...")
    OutputDTO_dict = {subset: []}
    for oid_batch in tqdm.tqdm(oid_batches):
        detections = all_detections.loc[oid_batch]
        features = all_features.loc[oid_batch]

        # InputDTO with defined batch size 
        input_dto = input_dto_factory(
            detections=detections, 
            non_detections=pd.DataFrame(), 
            features=features, 
            xmatch=pd.DataFrame(), 
            stamps=pd.DataFrame()
        )

        # Model
        model = SquidwardFeaturesClassifier(
            model_path=os.getenv("TEST_SQUIDWARD_MODEL_PATH"),
            mapper=SquidwardMapper(),
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
    ROOT = "../../../data_acquisition/ztf_forced_photometry"

    dir_astro_lightcurves = "{}/preprocessed/data_241209_ao".format(ROOT)
    dir_astro_features = "{}/preprocessed/data_241209_ao_shorten_features".format(ROOT)
    path_partition = '{}/preprocessed/partitions/241209/partitions.parquet'.format(ROOT)

    batch_size = 50
    subset = 'test' # ['training', 'validation', 'test']

    use_multiprocessing = True
    num_cores = 20

    run(dir_astro_lightcurves,
        dir_astro_features, 
        path_partition,
        subset,
        batch_size,
        use_multiprocessing,
        num_cores)