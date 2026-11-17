"""Cross-match of a batch of messages: service query, attach to the messages, scribe commands."""
from typing import Any, Dict, List  # type hints of the methods


def _match_for_output(match: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a service match for the output topic: keep only the map-valued
    `metadata` entries.

    The xmatch service returns each metadata row with its own `id` column, a
    plain string equal to `match_id`. The output schema
    (`schemas/feature_ms_step/ztf/xmatch.avsc`) declares `metadata` as a map of
    maps, so any scalar entry makes Avro serialization fail. Nothing downstream
    reads the scalars: the parser only uses the `w*mpro` maps. The scribe path
    receives the untouched match list, not this copy.
    """
    metadata = match.get("metadata") or {}  # None or missing counts as empty
    return {  # copy of the match with the filtered metadata
        **match,  # every other key as received
        "metadata": {k: v for k, v in metadata.items() if isinstance(v, dict)},  # map-valued entries only
    }  # end of the shaped match


class XmatchEnricher:
    """Query the xmatch service for a batch, attach the allwise match to each message and build the scribe commands.

    Parameters
    ----------
    client : XmatchClient
        Client of the xmatch service (anything with `conesearch_with_metadata`).
    catalogs : list
        Catalogs to cross-match against, one conesearch request per catalog.
    survey_name : str
        Survey name written in each scribe command.
    """

    def __init__(self, client, catalogs: list, survey_name: str):
        """Keep the client, the catalogs and the survey name."""
        self.client = client  # service client; the step may replace it before a batch
        self.catalogs = catalogs  # one request per catalog
        self.survey_name = survey_name  # written in every command

    def query(self, messages: List[dict]) -> List[Dict[str, Any]]:
        """
        Query the xmatch service for every message of the batch, whatever the survey,
        skipping the messages whose `sid` is 2, with one conesearch request per
        catalog in `catalogs`.

        Parameters
        ----------
        messages : List[dict]
            List of messages containing oid, ra, dec information

        Returns
        -------
        List[Dict[str, Any]]
            List of xmatch results, each containing oid and match information
        """
        oids = []  # object ids to search, as strings
        ras = []  # right ascensions, same order
        decs = []  # declinations, same order

        for msg in messages:  # every incoming message
            if msg['sid'] == 2:  # messages of sid 2 are not cross-matched
                continue  # Saltar sid 2
            oids.append(str(msg["oid"]))  # oid as string
            ras.append(msg["meanra"])  # mean right ascension of the object
            decs.append(msg["meandec"])  # mean declination of the object

        # One request per catalog: the server scopes the search to each catalog
        # before the KNN, so we get the nearest match in every catalog instead
        # of only the single global nearest.
        results = []  # matches of every catalog
        for catalog in self.catalogs:  # one request per catalog
            results += self.client.conesearch_with_metadata(  # nearest match per object in this catalog
                ras=ras,  # right ascensions
                decs=decs,  # declinations
                oids=oids,  # object ids
                catalog=catalog,  # catalog to search
            )  # end of the request

        return results  # one match per (oid, catalog) found

    def attach(self, messages: List[dict], results: List[Dict[str, Any]]) -> None:
        """Set `msg["xmatches"]` on every message to its allwise match shaped for the output, or None."""
        # results holds one match per (oid, catalog). Features only
        # use the allwise match (W1-W4), so map oid -> allwise match for the
        # per-message attach; the full list still goes to scribe.
        xmatch_dict = {  # oid as string -> allwise match shaped for the output
            str(match['oid']): _match_for_output(match)  # scalar metadata entries dropped
            for match in results  # every match
            if match.get('catalog') == 'allwise'  # allwise only
        }  # end of the allwise matches

        for msg in messages:  # every incoming message
            oid_str = str(msg['oid'])  # oid as string, like the matches carry it
            if oid_str in xmatch_dict:  # the object has an allwise match
                msg['xmatches'] = xmatch_dict[oid_str]  # attach it
            else:  # no allwise match
                msg['xmatches'] = None  # explicit None

    def commands(self, results: List[Dict[str, Any]], messages: List[dict]) -> List[Dict[str, Any]]:
        """Return one scribe xmatch command per match in `results`, in order; empty when there are no results."""
        # Crear mapeo de oid -> sid desde los mensajes
        oid_to_sid = {}  # oid as string -> sid of its message
        for msg in messages:  # every incoming message
            oid = str(msg['oid'])  # oid as string, like the matches carry it
            sid = msg['sid']  # survey id of the message
            oid_to_sid[oid] = sid  # last message of an oid wins

        # Procesar cada resultado de xmatch
        commands = []  # one command per match
        for match in results:  # every match
            oid = str(match["oid"])  # oid of the match as string
            sid = oid_to_sid.get(oid, 1)  # Usar sid del mensaje o valor por defecto
            commands.append({  # scribe xmatch command
                'step': 'xmatch',  # scribe step
                'survey': self.survey_name,  # survey name
                'payload': {  # row to write
                    'oid': int(oid),  # object id
                    'sid': sid,  # survey id
                    'catalog': match.get('catalog', 'allwise'),  # catalog of the match
                    'dist': match.get('distance'),  # distance to the match
                    'oid_catalog': match.get('match_id')  # id of the match in its catalog
                }  # end of the payload
            })  # end of the command
        return commands  # commands in the order of the matches
