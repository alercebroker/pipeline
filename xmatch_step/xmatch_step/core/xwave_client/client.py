import pandas as pd
import aiohttp
import asyncio
import time
import numpy as np


class XwaveClient:
    def __init__(self, base_url):
        self.base_url = base_url


    def execute(
        self,
        catalog,
        catalog_type: str,
        ext_catalog: str,
        ext_columns: list,
        selection: int,
        result_type: str,
        distmaxarcsec: int = 1.005, # To account for slight differences with CDS we use a slightly bigger radius
    ):
        """ 
            catalog: input data for the xmatch (OID, ra, dec)
            catalog_type: data type of the catalof (for xwave is pandas)
                # not in use
            ext_catalog: an allias for the name of the catalog
                # not in use
            ext_columns: parameter cols2 of cds request. It select wich of the
                available columns to return. default none, if none return all data.
            selection: n neighbors for xwave request
            result_type: data type for the return of this function (for xwave is pandas)
                # not in use
            distmaxarcsec: the search radious
        """

        async def async_execute():
            batches = self.split_dataframe(catalog, num_parts=5)

            async with aiohttp.ClientSession() as session:
                tasks = []
                for batch in batches:
                    for index, row in batch.iterrows():
                        # with this we have a list of dict with 
                        # id_in, ra_in, dec_in, RAJ2000, DEJ2000, ALLWISE
                        ra = row['ra']
                        dec = row['dec']
                        oid = row['oid']
                        task = asyncio.create_task(self.process_coordinate(session, ra, dec, oid, selection, distmaxarcsec))
                        tasks.append(task)
                    results = await asyncio.gather(*tasks)
            result_coordinates_df = pd.DataFrame(results)
            batches_2 = self.split_dataframe(result_coordinates_df, num_parts=5)
            # otro proceso async para pedir metadada, pero necesita el primero
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for batch in batches_2:
                    for index, row in batch.iterrows():
                        # with this we have a list of dict with 
                        # ALLWISE, and all the metadara required
                        allwise_id = row['AllWISE']
                        task = asyncio.create_task(self.process_metadata(session, allwise_id, ext_columns))
                        tasks.append(task)
                    results = await asyncio.gather(*tasks)
            
            result_metadata_df = pd.DataFrame(results) 
            
            # hacer join de result coordinate y result metadata df
            result_complete_df = result_coordinates_df.merge(result_metadata_df, left_on="AllWISE", right_on="AllWISE")
            
            # Calcular la distancia angular (de momento se calcula en step. Más adelante puede venir desde el endpoint)
            result_complete_df = self.add_distance_column(result_complete_df)
            
            # Apply transformatiosn
            result_complete_df = self.apply_dataframe_transformations(result_complete_df)
            return result_complete_df
        
        return asyncio.run(async_execute())
    
    def split_dataframe(self, df, num_parts):
        batch_size = len(df) // num_parts
        return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]


        # Function to handle the request for each coordinate asynchronously
    async def process_coordinate(self, session, ra, dec, oid, selection, distmaxarcsec):
        """
        session: aiohttp async session oject
        ra: ra to search
        dec: dec to search
        oid: alerce oid of the object to search
        return: a list with a dict with the response 
        """
        url = f"{self.base_url}/v1/conesearch?ra={ra}&dec={dec}&radius={distmaxarcsec}&nneighbor={selection}"
        async with session.get(url) as response:
            if response.status == 200:
                # Parse the response
                data = await response.json()
                data = data[0]

                if data:
                    result_dict = {}                
                    result_dict['ra_in'] = ra
                    result_dict['dec_in'] = dec
                    result_dict['id_in'] = oid
                    result_dict['RAJ2000'] = data['Ra']
                    result_dict['DEJ2000'] = data['Dec']
                    result_dict['AllWISE'] = data['ID']
                return result_dict
            else:
                print(f"Failed to fetch data for ra={ra}, dec={dec}. Status code: {response.status}")
                return []

    async def process_metadata(self, session, allwise_id, projection=None):
        """
        session: aiohttp async session oject
        akkwise_id: allwise id to search
        return: a list with a dict with the response 
        """

        url = f"{self.base_url}/v1/metadata?id={allwise_id}&catalog=allwise"
        async with session.get(url) as response:
            if response.status == 200:
                # Parse the response
                data = await response.json()
                data = data[0]

                if data:
                    result_dict = {}
                    result_dict['AllWISE'] = allwise_id
                    for key, value in data.items():
                        if projection is None:
                            result_dict[key] = value
                        elif key in projection:
                            result_dict[key] = value
                return result_dict
            else:
                print(f"Failed to fetch metadata for id {allwise_id}. Status code: {response.status}")
                return []
                



    def haversine_distance(self, ra1, dec1, ra2, dec2):
        """
        Calculate angular distance between two points using haversine formula.
        Returns:
            Distance in arcseconds
        """
        # Convert to radians (necessary for Haversine)
        ra1_rad = np.radians(ra1)
        dec1_rad = np.radians(dec1)
        ra2_rad = np.radians(ra2)
        dec2_rad = np.radians(dec2)
        
        # Haversine formula
        delta_ra = ra2_rad - ra1_rad
        delta_dec = dec2_rad - dec1_rad
        
        a = (np.sin(delta_dec/2)**2 + 
            np.cos(dec1_rad) * np.cos(dec2_rad) * 
            np.sin(delta_ra/2)**2)
        
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        # Convert answer to arcseconds
        return float(c * 180.0 / np.pi * 3600.0)

    def add_distance_column(self, df):
        """
        Adds angDist column to the dataframe.
        angDist is calculated using haversine distance between input and output coordinates.
        """
        distances = []
        
        for index, row in df.iterrows():
            dist = self.haversine_distance(
                row['RAJ2000'], row['DEJ2000'],
                row['ra_in'], row['dec_in']
            )
            distances.append(dist)
        
        df['angDist'] = distances
        
        return df

    def reorder_dataframe(self, df):
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

    def rename_columns(self, df):
        """
        Apply column rename so the client response will be the same as the original XMatch client
        """
        rename_dict = {
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
        df = df.rename(columns=rename_dict)
        return df

    def apply_dataframe_transformations(self, df):
        """
        Function that applies all the tranformations to the dataframe, including removing the extra id column, renmaing, reordering and adding the incremental col
        """
        df['col1'] = range(len(df))
        df = df.drop('id', axis=1)    
        df = self.rename_columns(df)
        df = self.reorder_dataframe(df)
        return df
    


    ####################### v2 of client:

    """
    async def process_coordinate(session, ra, dec, oid, selection, distmaxarcsec):
    # Perform the conesearch query
    url = f"{self.base_url}/v1/conesearch?ra={ra}&dec={dec}&radius={distmaxarcsec}&nneighbor={selection}"
    async with session.get(url) as response:
        if response.status == 200:
            # Parse the conesearch response
            data = await response.json()
            results = []
            if data:
                for entry in data:
                    entry['ra_in'] = ra
                    entry['dec_in'] = dec
                    entry['id_in'] = oid
                    
                    # Query the client for metadata as soon as the conesearch data is available
                    metadata_url = f"{self.base_url}/v1/metadata?id={allwise_id}&catalog=allwise"
                    async with session.get(metadata_url) as metadata_response:
                        if metadata_response.status == 200:
                            metadata = await metadata_response.json()
                            # Combine the conesearch data with the metadata
                            entry.update(metadata)
                            results.append(entry)
                        else:
                            print(f"Failed to fetch metadata for ID={entry['ID']}. Status code: {metadata_response.status}")
            return results
        else:
            print(f"Failed to fetch data for ra={ra}, dec={dec}. Status code: {response.status}")
            return []

    
    
    """