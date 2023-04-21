# Previous Detections Step

Parses the `extra_fields` of the incoming detections to retrieve the binary field `prv_candidates` if the alert comes from ZTF. Otherwise, it does nothing.
This step returns a list with the processed objects, including their `aid`, previous candidates, detections without the `extra_fields` field and their non_detections.

## Code architecture

The step calls all the logic from the **core** folder, which contains all the main logic of the step.

## Development

By default, unrecognized surveys are considered to have only the incoming alert as a detection and do not have
any non-detections. 

To include new strategies, a new module must be included in `core.strategy`, whose name must match the lower case 
name of the survey in question. This must have a function called `extract_detections_and_non_detections`, which receives
the incoming alert. Other functions or classes are possible. The aforementioned function must return:

```python
# The values refer to the types of the return
{"aid": str, "detections": list, "non_detections": list}
```

**Important:** For everything to work, remember to import the module in the `__init__.py` file of `core.strategy`.
