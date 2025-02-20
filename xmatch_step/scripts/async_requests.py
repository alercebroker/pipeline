import pandas as pd
import aiohttp
import asyncio
import time
import numpy as np

# Read the CSV file
df = pd.read_csv('df_30k_oid.csv')

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


# Function to handle the request for each coordinate asynchronously
async def process_coordinate(session, ra, dec, oid):
    # Perform the conesearch query
    url = f"http://localhost:8080/v1/conesearch?ra={ra}&dec={dec}&radius=1&nneighbor=10"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            results = []
            if data:
                for entry in data:
                    entry['ra_in'] = ra
                    entry['dec_in'] = dec
                    entry['id_in'] = oid
                    
                    metadata_url = f"http://localhost:8080/v1/metadata?id={entry['ID']}&catalog=allwise"
                    async with session.get(metadata_url) as metadata_response:
                        if metadata_response.status == 200:
                            metadata = await metadata_response.json()
                            entry.update(metadata)
                            results.append(entry)
                        else:
                            print(f"Failed to fetch metadata for ID={entry['ID']}. Status code: {metadata_response.status}")
            return results
        else:
            print(f"Failed to fetch data for ra={ra}, dec={dec}. Status code: {response.status}")
            return []

# Function to process the entire dataframe asynchronously
async def process_dataframe(df):
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_coordinate(session, row['ra'], row['dec'], row['oid'])
            for _, row in df.iterrows()
        ]
        all_results = await asyncio.gather(*tasks)
        return [entry for sublist in all_results for entry in sublist]


async def main():
    all_results = await process_dataframe(df)
    results_df = pd.DataFrame(all_results)
    results_df = apply_dataframe_transformations(results_df)
    results_df.to_csv('crosswave_30k_results_async_2.csv', index=False)
    print("Data saved to crosswave_30k_results_async_2.csv")


num_executions = 10
execution_times_list = []
for i in range(num_executions):
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times_list.append(execution_time)
    print(f"Total execution time: {execution_time:.3f} seconds")
average_time = sum(execution_times_list) / len(execution_times_list)
print('The average time of execution to (async) Crosswave request is: ', str(average_time))