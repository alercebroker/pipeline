import pandas as pd
import aiohttp
import asyncio
import time
import numpy as np

class XwaveClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.num_workers = 5  # Configurable number of metadata workers

    def execute(
        self,
        catalog,
        catalog_type: str = None,
        ext_catalog: str = None,
        ext_columns: list = None,
        selection: int = 1,
        result_type: str = None,
        distmaxarcsec: float = 1.005,
    ):
        return asyncio.run(self.async_execute(
            catalog, ext_columns, selection, distmaxarcsec
        ))

    async def async_execute(self, catalog, ext_columns, selection, distmaxarcsec):
        metadata_queue = asyncio.Queue()
        results = []

        async with aiohttp.ClientSession() as session:
            # Start metadata workers
            workers = []
            for _ in range(self.num_workers):
                worker = asyncio.create_task(
                    self.metadata_worker(session, metadata_queue, results, ext_columns)
                )
                workers.append(worker)

            # Process coordinates
            coordinate_tasks = []
            for index, row in catalog.iterrows():
                task = self.process_single_coordinate(
                    session,
                    row['ra'],
                    row['dec'],
                    row['oid'],
                    metadata_queue,
                    selection,
                    distmaxarcsec
                )
                coordinate_tasks.append(task)

            total_results = sum(await asyncio.gather(*coordinate_tasks))

            # Signal workers to finish
            for _ in range(self.num_workers):
                await metadata_queue.put(None)

            # Wait for all workers to complete
            await asyncio.gather(*workers)

        # Create final dataframe and apply transformations
        if results:
            result_df = pd.DataFrame(results)
            result_df = self.apply_dataframe_transformations(result_df)
            return result_df
        return pd.DataFrame()

    async def metadata_worker(self, session, queue, results, projection):
        while True:
            try:
                entry = await queue.get()
                if entry is None:
                    queue.task_done()
                    break

                result = await self.process_metadata(session, entry, projection)
                if result:
                    results.append(result)
                queue.task_done()
            except Exception as e:
                print(f"Error in metadata worker: {str(e)}")
                queue.task_done()

    async def process_single_coordinate(self, session, ra, dec, oid, metadata_queue, selection, distmaxarcsec):
        url = f"{self.base_url}/v1/conesearch?ra={ra}&dec={dec}&radius={distmaxarcsec}&nneighbor={selection}"
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
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

    async def process_metadata(self, session, entry, projection=None):
        try:
            allwise_id = entry['ID']
            url = f"{self.base_url}/v1/metadata?id={allwise_id}&catalog=allwise"
            async with session.get(url) as response:
                if response.status == 200:
                    metadata = await response.json()
                    if metadata:
                        metadata = metadata[0]  # Extract first item from metadata response
                        result_dict = {**entry}  # Start with coordinate search results
                        for key, value in metadata.items():
                            if projection is None or key in projection:
                                result_dict[key] = value
                        return result_dict
                else:
                    print(f"Failed to fetch metadata for ID={allwise_id}. Status code: {response.status}")
                    return None
        except Exception as e:
            print(f"Error processing ID={entry['ID']}: {str(e)}")
            return None

    def haversine_distance(self, ra1, dec1, ra2, dec2):
        """Calculate angular distance between two points using haversine formula."""
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

    def add_distance_column(self, df):
        """Add angDist column to the dataframe."""
        distances = []
        
        for index, row in df.iterrows():
            dist = self.haversine_distance(
                row['RAJ2000'], row['DEJ2000'],
                row['ra_in'], row['dec_in']
            )
            distances.append(dist)
        
        df['angDist'] = distances
        return df

    def rename_columns(self, df):
        """Apply column rename for consistency with original XMatch client."""
        rename_dict = {
            'Ra': 'RAJ2000',
            'Dec': 'DEJ2000',
            'ID': 'AllWISE',
            'w1mpro': 'W1mag',
            'w2mpro': 'W2mag',
            'w3mpro': 'W3mag',
            'w4mpro': 'W4mag',
            'w1sigmpro': 'e_W1mag',
            'w2sigmpro': 'e_W2mag',
            'w3sigmpro': 'e_W3mag',
            'w4sigmpro': 'e_W4mag',
            'J_m_2mass': 'Jmag',
            'J_msig_2mass': 'e_Jmag',
            'H_m_2mass': 'Hmag',
            'H_msig_2mass': 'e_Hmag',
            'K_m_2mass': 'Kmag',
            'K_msig_2mass': 'e_Kmag'
        }
        return df.rename(columns=rename_dict)

    def reorder_dataframe(self, df):
        """Reorder columns to match original XMatch client."""
        desired_order = ['angDist', 'col1', 'id_in', 'ra_in', 'dec_in', 'AllWISE', 'RAJ2000',
                        'DEJ2000', 'W1mag', 'W2mag', 'W3mag', 'W4mag', 'Jmag', 'Hmag', 'Kmag',
                        'e_W1mag', 'e_W2mag', 'e_W3mag', 'e_W4mag', 'e_Jmag', 'e_Hmag',
                        'e_Kmag'
                        ]
        return df[desired_order]

    def apply_dataframe_transformations(self, df):
        """Apply all transformations to the dataframe."""
        df['col1'] = range(len(df))
        # Drop unnecessary columns
        columns_to_drop = ['Ipix', 'Cat'] if 'Cat' in df.columns else []
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            
        df = self.rename_columns(df)
        df = self.add_distance_column(df)
        df = self.reorder_dataframe(df)
        return df