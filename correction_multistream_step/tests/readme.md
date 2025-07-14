# Data conflict when there is same oid and measurement_id

Sometimes we'll have objects with same oid and measurement id between the data from the database and the messages, however there can only be one detection/forced photometry per oid/measurement_id. This means that we'll have deduplication of the data. We must then check that deduplication runs correctly in the integration test. The same occurs for non_detections, where there can only be one per oid and mjd.

On json files ond this same folder, for example 'data_input_prc_candidates_staging.json' we have some data and there is a conflict on the lines 5-6. There we have "oid": 1111111111 and "measurement_id": 0. This means that the data will be deduplicated during the lightcurve part of the step since we can only have one detection per oid/measurement_id, and we will only keep one of those detections for the rest of the process. 

In the test_utils file we test the deduplication of the data in the step as well as other tests. To do this first we insert the data from the data2.py file into a postgres db. Then we use the test_consumer and test producer to execute the step without going through kafka, using as input messages the ones in the data_input_prv_candidates_staging.json file. Since we know all the data inside both the data in the database and the messages, we can know exactly how many detections and non detections will be deduplicated and end up in the final results. 

We test the expected deduplication using two different oids. 

For the oid 1111111111:

    Detections: 

        In the messages in the json this oid has the following detections:
            - oid 1111111111 and mid 0 twice, in lines 5 and 6
            - oid 1111111111 and mid 1 in line 7 
            - oid 1111111111 and mid 2  in line 8

        In the database data we have the following detections and forced phots:
            - oid 1111111111 and mid 0 in line 73
            - oid 1111111111 and mid 0 in line 81
            - oid 1111111111 and mid 1 in line 410
            - oid 1111111111 and mid 2 in line 402 

        Thus we expect to have 3 deduplicated detections because the unique mids are 0, 1 and 2

    Non-detections: 
        
        In the messages in the json this oid has the following non-detections:
            - oid 1111111111 and mjd 60726.0 in line 10
            - oid 1111111111 and mjd 60727.0 in line 11 
            - oid 1111111111 and mjd 60728.0 twice, in line 12 and 13 
        
        In the database data we have the following non-detections:
            - oid 1111111111 and mjd 60726.0 in line 225
            - oid 1111111111 and mjd 60728.0 in line 226

        Thus we expect to have 3 deduplicated non-detections because there's only 3 unique mjds which are 60726.0, 60727.0 and 60728.0

        

For the oid 2222222222:
    
    Detections:

        In the messages in the json this oid has the following detections:
            - oid 2222222222 and mid 4 in line 18
            - oid 2222222222 and mid 5 in line 19
            - oid 2222222222 and mid 6 in line 20
            - oid 2222222222 and mid 7 in line 21

        In the database data we have the following detections:
            - oid 2222222222 and mid 2 in line 65-66 
            - oid 2222222222 and mid 2 in line 120-121 (ztf)

    Ztf_detections:
    
        In the database data we have the following forced phots:
            - oid 2222222222 and mid 10 in line 269-270 (ztf)
            - oid 2222222222 and mid 10 in line 394-395 

    Non_detections:

        In the messages in the json this oid has the following non detections:

            - oid 2222222222 and mjd 60726.0 line 23
            - oid 2222222222 and mjd 60727.0 line 24
            - oid 2222222222 and mjd 60728.0 line 25
            - oid 2222222222 and mjd 60720.0 line 26
        
        In the database data we have the following non detections:

            - oid 2222222222 and mjd 60679.0 line 223
            - oid 2222222222 and mjd 60678.0 line 224

