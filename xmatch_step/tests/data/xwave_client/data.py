import pandas as pd

# Input catalog data columns
# tiene oid, ra y dec
# 5 filas


def set_data_test():
    input_catalog = pd.DataFrame(
        {
            "oid": ["oid1", "oid2", "oid3", "oid4", "oid5"],
            "ra": [37.43285, 71.60753, 6.83195, 83.64525, 35.46444],
            "dec": [-1.69134, -8.42896, 83.85722, -81.44048, 63.26527],
        }
    )

    # Jsons de la primera request
    conesearch_responses = {
        "conesearch?ra=37.43285&dec=-1.69134&radius=1.005&nneighbor=1": [
            {
                "ID": "0378m016_ac51-015624",
                "Ipix": 297622515186,
                "Ra": 37.4329792,
                "Dec": -1.6914399,
                "Cat": "allwise",
            }
        ],
        "conesearch?ra=71.60753&dec=-8.42896&radius=1.005&nneighbor=1": [
            {
                "ID": "0719m091_ac51-057097",
                "Ipix": 379791380034,
                "Ra": 71.6073447,
                "Dec": -8.4288395,
                "Cat": "allwise",
            }
        ],
        "conesearch?ra=6.83195&dec=83.85722&radius=1.005&nneighbor=1": [
            {
                "ID": "0116p832_ac51-060705",
                "Ipix": 68352995551,
                "Ra": 6.8337888,
                "Dec": 83.8572158,
                "Cat": "allwise",
            }
        ],
        "conesearch?ra=83.64525&dec=-81.44048&radius=1.005&nneighbor=1": [
            {
                "ID": "0875m818_ac51-038244",
                "Ipix": 550912614293,
                "Ra": 83.6442345,
                "Dec": -81.4403045,
                "Cat": "allwise",
            }
        ],
        "conesearch?ra=35.46444&dec=63.26527&radius=1.005&nneighbor=1": [
            {
                "ID": "0363p636_ac51-014909",
                "Ipix": 61313018866,
                "Ra": 35.4645235,
                "Dec": 63.2653672,
                "Cat": "allwise",
            }
        ],
    }

    # Jsons de la segunda request (si es que la cosa se divide en dos requests al final)
    metadata_responses = {
        "metadata?id=0378m016_ac51-015624&catalog=allwise": 
            {
                "id": "0378m016_ac51-015624",
                "W1mpro": 16.882,
                "W1sigmpro": 0.092,
                "W2mpro": 16.765,
                "W2sigmpro": None,
                "W3mpro": 12.126,
                "W3sigmpro": None,
                "W4mpro": 8.892,
                "W4sigmpro": None,
                "J_m_2mass": None,
                "J_msig_2mass": None,
                "H_m_2mass": None,
                "H_msig_2mass": None,
                "K_m_2mass": None,
                "K_msig_2mass": None,
            }
        ,
        "metadata?id=0719m091_ac51-057097&catalog=allwise": 
            {
                "id": "0719m091_ac51-057097",
                "W1mpro": 17.359,
                "W1sigmpro": 0.115,
                "W2mpro": 16.881,
                "W2sigmpro": None,
                "W3mpro": 11.883,
                "W3sigmpro": None,
                "W4mpro": 7.655,
                "W4sigmpro": None,
                "J_m_2mass": None,
                "J_msig_2mass": None,
                "H_m_2mass": None,
                "H_msig_2mass": None,
                "K_m_2mass": None,
                "K_msig_2mass": None,
            }
        ,
        "metadata?id=0116p832_ac51-060705&catalog=allwise": 
            {
                "id": "0116p832_ac51-060705",
                "W1mpro": 15.536,
                "W1sigmpro": 0.037,
                "W2mpro": 15.355,
                "W2sigmpro": 0.065,
                "W3mpro": 13.049,
                "W3sigmpro": None,
                "W4mpro": 9.346,
                "W4sigmpro": None,
                "J_m_2mass": 16.636,
                "J_msig_2mass": 0.14,
                "H_m_2mass": 15.784,
                "H_msig_2mass": 0.199,
                "K_m_2mass": 15.728,
                "K_msig_2mass": 0.252,
            }
        ,
        "metadata?id=0875m818_ac51-038244&catalog=allwise": 
            {
                "id": "0875m818_ac51-038244",
                "W1mpro": 14.354,
                "W1sigmpro": 0.026,
                "W2mpro": 14.381,
                "W2sigmpro": 0.035,
                "W3mpro": 13.198,
                "W3sigmpro": None,
                "W4mpro": 9.521,
                "W4sigmpro": 0.491,
                "J_m_2mass": 15.353,
                "J_msig_2mass": 0.051,
                "H_m_2mass": 14.577,
                "H_msig_2mass": 0.078,
                "K_m_2mass": 14.563,
                "K_msig_2mass": 0.096,
            }
        ,
        "metadata?id=0363p636_ac51-014909&catalog=allwise": 
            {
                "id": "0363p636_ac51-014909",
                "W1mpro": 13.429,
                "W1sigmpro": 0.026,
                "W2mpro": 13.464,
                "W2sigmpro": 0.031,
                "W3mpro": 12.098,
                "W3sigmpro": 0.322,
                "W4mpro": 9.179,
                "W4sigmpro": None,
                "J_m_2mass": 14.047,
                "J_msig_2mass": 0.035,
                "H_m_2mass": 13.75,
                "H_msig_2mass": 0.038,
                "K_m_2mass": 13.596,
                "K_msig_2mass": 0.043,
            }
        ,
    }

    output_catalog = pd.DataFrame(
        {
            "angDist": [
                0.5877831847809013,
                0.7896941579451099,
                0.7085101564648748,
                0.8338111072863096,
                0.37514073290711863,
            ],  # Random for now, must add the true distance measured later
            "col1": [0, 1, 2, 3, 4],  # Incremental values
            "oid_in": [
                "oid1",
                "oid2",
                "oid3",
                "oid4",
                "oid5",
            ],  # List of oids sent in bt the request
            "ra_in": [
                37.43285,
                71.60753,
                6.83195,
                83.64525,
                35.46444,
            ],  # Ra of the conesearch
            "dec_in": [
                -1.69134,
                -8.42896,
                83.85722,
                -81.44048,
                63.26527,
            ],  # Dec of the conesearch
            "AllWISE": [
                "0378m016_ac51-015624",
                "0719m091_ac51-057097",
                "0116p832_ac51-060705",  # Identifier of allwise source id (unique)
                "0875m818_ac51-038244",
                "0363p636_ac51-014909",
            ],
            "RAJ2000": [
                37.4329792,
                71.6073447,
                6.8337888,
                83.6442345,
                35.4645235,
            ],  # Ra of the catalog source
            "DEJ2000": [
                -1.6914399,
                -8.4288395,
                83.8572158,
                -81.4403045,
                63.2653672,
            ],  # Dec of the catalog source
            "W1mag": [
                16.882,
                17.359,
                15.536,
                14.354,
                13.429,
            ],  # Metadata columns
            "W2mag": [16.765, 16.881, 15.355, 14.381, 13.464],
            "W3mag": [12.126, 11.883, 13.049, 13.198, 12.098],
            "W4mag": [8.892, 7.655, 9.346, 9.521, 9.179],
            "Jmag": [None, None, 16.636, 15.353, 14.047],
            "Hmag": [None, None, 15.784, 14.577, 13.750],
            "Kmag": [None, None, 15.728, 14.563, 13.596],
            "e_W1mag": [0.092, 0.115, 0.037, 0.026, 0.026],
            "e_W2mag": [None, None, 0.065, 0.035, 0.031],
            "e_W3mag": [None, None, None, None, 0.322],
            "e_W4mag": [None, None, None, 0.491, None],
            "e_Jmag": [None, None, 0.140, 0.051, 0.035],
            "e_Hmag": [None, None, 0.199, 0.078, 0.038],
            "e_Kmag": [None, None, 0.252, 0.096, 0.043],
        }
    )

    return (
        input_catalog,
        conesearch_responses,
        metadata_responses,
        output_catalog,
    )


client_parameters = {
    "catalog_type": None,
    "ext_catalog": None,
    "ext_columns": None,
    "selection": 1,
    "result_type": None,
    "distmaxarcsec": 1.005,
}
