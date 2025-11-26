from typing import List, Dict, Tuple

def update_object(messages: List[dict], objstats_dict: List[dict]) -> List[dict]:

    object_needs_update = []
    
    for message in messages:
        message_oid = message.get('oid')
        
        # Find the corresponding objectstats for this message's OID
        matching_objstats = objstats_dict.get(message_oid)
        if matching_objstats is None:
            continue  # Skip if no matching objectstats found
            
        objstats_lastmjd = matching_objstats.get('lastmjd')
        if objstats_lastmjd is None:
            continue
            
        # Get sources from the message
        sources = message.get('sources', [])
        
        # Find the maximum mjd from all sources in this message
        max_mjd = max(source.get('mjd') for source in sources)
        
        # Check if max MJD is equal or higher than objstats lastmjd
        if max_mjd >= objstats_lastmjd:
            dia_object = message.get('dia_object')
            if dia_object is not None:
                # dia_object can be an array. We will extend the list with the two objects
                if isinstance(dia_object, list):
                    object_needs_update.extend(dia_object)
                else:
                    # when its not an array, simply append the object to the list
                    object_needs_update.append(dia_object)
    
    return object_needs_update