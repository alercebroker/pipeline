class ClassOrder:
    datasets_order_classes = {
        "atlas": [
            "SNIa", "SNIbc", "SNIIb", "SNII", "SNIIn", "SLSN", 
            "TDE", "Microlensing", "QSO", "AGN", "Blazar", 
            "YSO", "CV/Nova", "LPV", "EA", "EB/EW", 
            "Periodic-Other", "RSCVn", "CEP", "RRLab", "RRLc", "DSCT"
        ],
        "ztf_ff": [
            "SNIa", "SNIbc", "SNIIb", "SNII", "SNIIn", "SLSN", 
            "TDE", "Microlensing", "QSO", "AGN", "Blazar", 
            "YSO", "CV/Nova", "LPV", "EA", "EB/EW", 
            "Periodic-Other", "RSCVn", "CEP", "RRLab", "RRLc", "DSCT"
        ],
        "ztf_ff_sanchez_tax": [
            "SNIa", "SNIbc", "SNII", "SLSN",
            "QSO", "AGN", "Blazar",
            "YSO", "CV/Nova", "LPV", "E",
            "DSCT", "RRL", "CEP", "Periodic-Other", "Others"
        ],
    }

    @staticmethod
    def get_order(name_dataset):
        return ClassOrder.datasets_order_classes.get(name_dataset, [])