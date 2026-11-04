"""
coordinates: list (dic); lista de oid y coordenadas
    {oid: id_del_objeto, ra: right_ascention, dec: declination}

filter: dic; info para definir un filtro (clase, clasificador, umbral de probabilidad)
    {class: class_to_filter, clasifier: name_clasifier, prob: min_prob_acceptable}

calc_dispertion: bool. if True, delight will calculate the dispertion of the galaxy

calc_galaxy_size: bool. if True, delight will calculate the size of the galaxy

"""

def delight_processing(input_json):

    """
    Process input JSON data to extract information and realize calculations.

    :param dict input_json: Input JSON data containing coordinates and filter information.
    :return: Processed data including coordinates, dispersion, and galaxy size.
    :rtype: dict
    """

    # Extraer información del input
    coordinates = input_json.get("coordinates", [])
    calc_dispersion = input_json.get("calc_dispersion", False)
    calc_galaxy_size = input_json.get("calc_galaxy_size", False)

    # Procesar los datos
    output_data = process_data(coordinates, calc_dispersion, calc_galaxy_size)
    return output_data


def process_data(coordinates, calc_dispersion, calc_galaxy_size):
    """
    Process input coordinates and realize calculations if required.

    :param list coordinates: List of dictionaries containing coordinate information.
    :param dict filter_info: Filter information.
    :param bool calc_dispersion: Flag indicating whether to calculate dispersion.
    :param bool calc_galaxy_size: Flag indicating whether to calculate galaxy size.
    :return: Processed data including coordinates, dispersion, and galaxy size.
    :rtype: dict
    """
    
    # Process data debe retornar en coordinates oid, ra, dec, ra_gal, dec_gal 
    # Devolver la misma lista de coordenadas como output
    output_coordinates = [
        {
            "oid": input_coordinates["oid"],
            "ra": input_coordinates["ra"],
            "dec": input_coordinates["dec"],
            "ra_gal": get_galaxy_coordinates(input_coordinates)["ra"],
            "dec_gal": get_galaxy_coordinates(input_coordinates)["dec"],
            "dispersion": calculate_dispersion(coordinates) if calc_dispersion else None,
            "galaxy_size": calculate_galaxy_size(coordinates) if calc_galaxy_size else None,
        }
        for input_coordinates in coordinates
    ]

    return output_coordinates

# Pleaseholder
 
def calculate_dispersion(coordinates):
    # Implementacion pendiente
    return 1.0  #  Ejemplo

def calculate_galaxy_size(coordinates):
    # Implementacion pendiente
    return 42.0  # Ejemplo

def get_galaxy_coordinates(coordinates):
    # Implementacion pendiente
    return {"ra": 12.34, "dec": -45.67}  # Ejemplo
