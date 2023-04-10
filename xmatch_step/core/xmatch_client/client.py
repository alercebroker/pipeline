import io
import sys
import pandas as pd
from astropy.table import Table
import astropy.io.votable as votable
import requests
import json
import pkg_resources

# catalog alias
input_path = pkg_resources.resource_filename(
    "cds_xmatch_client", "data/catalog_alias.json"
)
with open(input_path) as catalogs_file:
    CATALOG_MAP = json.load(catalogs_file)


class XmatchClient:
    @staticmethod
    def execute(
            catalog,
            catalog_type: str,
            ext_catalog: str,
            ext_columns: list,
            selection: str,
            result_type: str,
            distmaxarcsec: int = 1,
    ):

        try:
            # catalog alias
            if ext_catalog in CATALOG_MAP:
                ext_catalog = CATALOG_MAP[ext_catalog]

            # Encode input
            if catalog_type == "pandas":

                string_io = io.StringIO()
                new_columns = {}
                for c in catalog.columns:
                    new_columns[c] = "%s_in" % c
                catalog.rename(columns=new_columns, inplace=True)

                catalog.to_csv(string_io)
                catalog_content = string_io.getvalue()

            elif catalog_type == "astropy":

                columns = list(catalog.columns)
                for c in columns:
                    catalog.rename_column(c, "%s_in" % c)

                bytes_io = io.BytesIO()
                catalog = votable.from_table(catalog)
                votable.writeto(catalog, bytes_io)
                catalog_content = bytes_io.getvalue()

            else:
                raise Exception("Unknown input type %s" % catalog_type)

            # columns
            ext_columns_str = None
            if ext_columns is not None:
                ext_columns_str = ",".join(ext_columns)

            # response format
            if result_type == "pandas":
                response_format = "csv"
            elif result_type == "astropy":
                response_format = "votable"
            else:
                raise Exception("Unknown output type %s" % result_type)

            # params
            params = {
                "request": "xmatch",
                "distMaxArcsec": distmaxarcsec,
                "selection": selection,
                "RESPONSEFORMAT": response_format,
                "cat2": ext_catalog,
                "colRA1": "ra_in",
                "colDec1": "dec_in",
            }
            if ext_columns_str is not None:
                params["cols2"] = ext_columns_str

            # Send the request
            r = requests.post(
                "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync",
                data=params,
                files={"cat1": catalog_content},
            )
            if not r.ok:
                raise Exception(f"CDS Request returned status {r.status_code}")

            # Decode output
            response_bytes = io.BytesIO(r.content)
            result = None
            if result_type == "pandas":
                result = pd.read_csv(response_bytes)

            elif result_type == "astropy":
                result = Table.read(response_bytes)

        except Exception as exception:

            sys.stderr.write("Request to CDS xmatch failed: %s \n" % exception)
            raise exception

        return result