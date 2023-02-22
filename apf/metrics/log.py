from apf.metrics import GenericMetricsProducer
import logging
import tempfile
import os


class LogfileMetricsProducer(GenericMetricsProducer):
    """Write metrics into a logfile

    Parameters
    ----------
    config : dict
        Allowed parameters:

        - "PATH": path to logfile, if not set a tempfile will be created in /tmp.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.metrics_logger = logging.Logger("metrics")

        path = self.config.get("PATH")
        if path is None:
            default_tmp_dir = tempfile._get_default_tempdir()
            temp_name = next(tempfile._get_candidate_names())
            name = self.__class__.__name__.lower()
            name = f"{name}-{temp_name}.log"
            path = os.path.join(default_tmp_dir, name)

        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        fh = logging.FileHandler(path)
        debug = self.config.get("DEBUG")
        if debug:
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)

        self.metrics_logger.addHandler(fh)
        self.logger.info(f"Writing metrics logs into {path}")

    def send_metrics(self, metrics):
        """Write metrics to logfile.

        Parameters
        ----------
        metrics : dict
            Metrics to be written.

        """
        metrics_str = ", ".join([f"{key}: {metrics[key]}" for key in metrics])
        self.metrics_logger.info(metrics_str)
