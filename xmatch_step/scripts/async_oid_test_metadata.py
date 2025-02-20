import pandas as pd
import aiohttp
import asyncio
import time
import numpy as np

distmaxarcsec = 1.005
selection = 1
num_parts = 1


def haversine_distance(ra1, dec1, ra2, dec2):
    """
    Calculate angular distance between two points using haversine formula.
    Returns:
        Distance in arcseconds
    """
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)
    
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad
    
    a = (np.sin(delta_dec/2)**2 + 
         np.cos(dec1_rad) * np.cos(dec2_rad) * 
         np.sin(delta_ra/2)**2)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return float(c * 180.0 / np.pi * 3600.0)

def add_distance_column(df):
    """
    Adds angDist column to the dataframe.
    angDist is calculated using haversine distance between input and output coordinates.
    """
    distances = []
    
    for index, row in df.iterrows():
        dist = haversine_distance(
            row['RAJ2000'], row['DEJ2000'],
            row['ra_in'], row['dec_in']
        )
        distances.append(dist)
    
    df['angDist'] = distances
    
    return df

def rename_columns(df):
    """
    Apply column rename so the client response will be the same as the original XMatch client
    """
    rename_dict = {
        'Ra': 'RAJ2000',
        'Dec': 'DEJ2000',
        'Source_id': 'AllWISE',
        'W1mpro': 'W1mag',
        'W2mpro': 'W2mag',
        'W3mpro': 'W3mag',
        'W4mpro': 'W4mag',
        'W1sigmpro': 'e_W1mag',
        'W2sigmpro': 'e_W2mag',
        'W3sigmpro': 'e_W3mag',
        'W4sigmpro': 'e_W4mag',
        'J_m_2mass': 'Jmag',
        'J_msig_2mass': 'e_Jmag',
        'H_m_2mass': 'Hmag',
        'H_msig_2mass': 'e_Hmag',
        'K_m_2mass': 'Kmag',
        'K_msig_2mass': 'e_Kmag'
    }
    df = df.rename(columns=rename_dict)
    return df

def reorder_dataframe(df):
    """
    Reorders the dataframe so the client response will be the same as the original XMatch client
    """
    desired_order = ['angDist', 'col1', 'id_in', 'ra_in', 'dec_in', 'AllWISE', 'RAJ2000',
                    'DEJ2000', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag', 'Hmag', 'Kmag',
                    'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag', 'e_Jmag', 'e_Hmag',
                    'e_Kmag'
                    ]
    df = df[desired_order]
    return df

def apply_dataframe_transformations(df):
    """
    Function that applies all the tranformations to the dataframe, including removing the extra id column, renmaing, reordering and adding the incremental col
    """
    df['col1'] = range(len(df))
    df = df.drop('ID', axis=1).drop('Ipix', axis=1).drop('Cat', axis=1)    
    df = rename_columns(df)
    df = add_distance_column(df)
    df = reorder_dataframe(df)
    return df


async def process_single_coordinate(session, ra, dec, oid, metadata_queue):
    url = f"http://localhost:8080/v1/conesearch?ra={ra}&dec={dec}&radius={distmaxarcsec}&nneighbor={selection}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data:
                    # Immediately queue metadata requests for each result
                    for entry in data:
                        entry['ra_in'] = ra
                        entry['dec_in'] = dec
                        entry['id_in'] = oid
                        await metadata_queue.put(entry)
                    return len(data)
            else:
                print(f"Failed to fetch data for ra={ra}, dec={dec}. Status code: {response.status}")   
            return 0
    except Exception as e:
        print(f"Error in coordinate search: {str(e)}")
        return 0

async def process_metadata(session, entry):
    try:
        url = f"http://localhost:8080/v1/metadata?id={entry['ID']}&catalog=allwise"
        async with session.get(url) as response:
            if response.status == 200:
                metadata = await response.json()
                return {**entry, **metadata}
            else:
                print(f"Failed to fetch metadata for ID={entry['ID']}. Status code: {response.status}")
                return None
    except Exception as e:
        print(f"Error processing ID={entry['ID']}: {str(e)}")
        return None

async def metadata_worker(session, queue, results):
    while True:
        try:
            entry = await queue.get()
            if entry is None: 
                queue.task_done()
                break
            
            result = await process_metadata(session, entry)
            if result:
                results.append(result)
            queue.task_done()
        except Exception as e:
            print(f"Error in metadata worker: {str(e)}")
            queue.task_done()

async def main():
    df = pd.read_csv('df_30k_oid.csv')
    print(f"Starting processing for {len(df)} entries...")
    
    metadata_queue = asyncio.Queue()
    results = []
    
    # Create a single session for all requests
    async with aiohttp.ClientSession() as session:
        num_workers = 1  
        workers = []
        for _ in range(num_workers):
            worker = asyncio.create_task(metadata_worker(session, metadata_queue, results))
            workers.append(worker)
        
        coordinate_tasks = []
        for index, row in df.iterrows():
            task = process_single_coordinate(
                session, 
                row['ra'], 
                row['dec'], 
                row['oid'], 
                metadata_queue
            )
            coordinate_tasks.append(task)
        
        
        total_results = sum(await asyncio.gather(*coordinate_tasks))
        print(f"Coordinate search complete. Found {total_results} results.")
        
        for _ in range(num_workers):
            await metadata_queue.put(None)
        
        
        await asyncio.gather(*workers)
    
    # Create final dataframe and save results
    final_df = pd.DataFrame(results)
    final_df = apply_dataframe_transformations(final_df)
    final_df.to_csv('crosswave_30k_complete.csv', index=False)
    print(f"\nProcess complete. Saved {len(final_df)} results.")
    return len(final_df)

# Measure performance
num_executions = 10
execution_times_list = []

for i in range(num_executions):
    print(f"\nExecution {i+1}/{num_executions}")
    start_time = time.time()
    results_count = asyncio.run(main())
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_list.append(execution_time)
    print(f"Total execution time: {execution_time:.3f} seconds")
    print(f"Total results processed: {results_count}")

average_time = sum(execution_times_list) / len(execution_times_list)
print(f'\nAverage execution time: {average_time:.3f} seconds')
