class ClassOrder:
    datasets_order_classes = {
        "ztf_ff": [
            "SNIa", "SNIbc", "SNIIb", "SNII", "SNIIn", "SLSN", 
            "TDE", "Microlensing", "QSO", "AGN", "Blazar", 
            "YSO", "CV/Nova", "LPV", "EA", "EB/EW", 
            "Periodic-Other", "RSCVn", "CEP", "RRLab", "RRLc", "DSCT"
        ],
        # Agrega aquí más datasets si es necesario
    }

    @staticmethod
    def get_order(name_dataset):
        return ClassOrder.datasets_order_classes.get(name_dataset, [])