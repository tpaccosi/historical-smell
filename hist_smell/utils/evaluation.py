import numpy as np


def results_to_json(results):
    json_results = {}
    for label in sorted(results):
        if label.startswith("overall_"):
            json_results[label] = float(results[label])
        else:
            json_results[label] = {}
            for measure in results[label]:
                if isinstance(results[label][measure], np.float64):
                    json_results[label][measure] = float(results[label][measure])
                elif isinstance(results[label][measure], np.int64):
                    json_results[label][measure] = int(results[label][measure])
    return json_results
