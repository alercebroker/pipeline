from typing import Dict, List, Tuple

from apf.core.step import GenericStep
from delight_step.clients import dummy_processing
from delight_step.database_sql import PSQLConnection, get_sql_probabily, only_unique_oid_parser

class DelightStep(GenericStep):
    """
    Resumen de lo que hace el step 
    """
    def __init__(
        self,
        config=None,
        db_client=None,
        **step_args,
    ):
        super().__init__(config=config, **step_args)
        # de config sacar db config
        # pasar el dic de psql al PSQLConnection y crear una conexion asignarlo a un self.db_client por ejemplo
        if db_client:
            self.db_client = db_client
        else:
            self.db_client = PSQLConnection(config.get("PSQL_CONFIG"))
        
        delight_config = self.config.get("DELIGHT_CONFIG", None)
        self.filter_info = delight_config.get("filter", None)
        self.calc_galaxy_size = delight_config.get("calc_galaxy_size", False)
        self.calc_dispersion = delight_config.get("calc_dispersion", False)

    def execute(self, messages: List[dict]): 
        #Primer paso: Crear el input del cliente delight a partir de "messages"

        # crear una lista con todos los oids en el input
        oids = [message["oid"] for message in messages]

        # hacer la consulta a db con get_stql_probabiliy para obtener los oids que
        # cumplan con el filtro usando el only oid parser
        filtered_oids = get_sql_probabily(
            oids=oids,
            db_sql=self.db_client,
            probability_filter=self.filter_info,
            parser=only_unique_oid_parser
        )
        print(f" oids = \n\t{oids}\n------\nfiltered = \n\t{filtered_oids}")

        coordinates = [
            {
                "oid": message["oid"],
                "ra": message["meanra"],
                "dec": message["meandec"]
            }
            for message in messages 
            if message["oid"] in filtered_oids
        ]
        
        # Objeto Python que representa el input del cliente delight
        delight_input_dict = {
            "coordinates": coordinates,
            "calc_dispersion": self.calc_dispersion,
            "calc_galaxy_size": self.calc_galaxy_size
        }
        delight_data = dummy_processing(delight_input_dict)
                
        #Tercer paso: parsing de output y produce a srcribe
        return self.parse_output(messages, delight_data)
    
    def parse_output(self, input_messages: List[dict], delight_data):
        """"
        {
            "oid": oid,
            "meanra": meanra,
            "meandec": meandec,
            "detections": [detections],
            "non_detections": [non_detections]
            "galaxy_properties": galaxy_properties
        }
        """
        def get_galaxy_property_from_oid(oid):
            return lambda x: x["oid"] == oid
        
        result = []

        # iterar por delight data

        for message in input_messages:
            delight_data_for_oid = list(filter(
                get_galaxy_property_from_oid(message["oid"]),
                delight_data
            ))
            object_output_dict= {
                "oid": message["oid"],
                "candid": message["candid"],
                "ra": message["meanra"],
                "dec": message["meandec"],
                "detections": message["detections"],
                "non_detections": message["non_detections"],
                "galaxy_properties": None if len(delight_data_for_oid) == 0 else {
                    "ra": delight_data_for_oid[0]["ra_gal"],
                    "dec": delight_data_for_oid[0]["dec_gal"],     
                    "dispersion": delight_data_for_oid[0]["dispersion"],
                    "galaxy_size": delight_data_for_oid[0]["galaxy_size"]               
                }
            }
        
            result.append(object_output_dict)

        return result

