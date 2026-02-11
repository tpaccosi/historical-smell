import json
import sys

pathTrain = sys.argv[1]
pathDev = sys.argv[2]

weights = [float(w) for w in sys.argv[3:13]]


task1Dict =  {
                "task_type": "seq_bio",
                "column_idx": 4,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[0]
            }

task2Dict =  {
                "task_type": "seq_bio",
                "column_idx": 5,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[1]
            }

task3Dict =  {
                "task_type": "seq_bio",
                "column_idx": 6,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[2]
            }

task4Dict =  {
                "task_type": "seq_bio",
                "column_idx": 7,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[3]
            }

task5Dict =  {
                "task_type": "seq_bio",
                "column_idx": 8,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[4]
            }

task6Dict =  {
                "task_type": "seq_bio",
                "column_idx": 9,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[5]
            }

task7Dict =  {
                "task_type": "seq_bio",
                "column_idx": 10,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[6]
            }


task8Dict =  {
                "task_type": "seq_bio",
                "column_idx": 11,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[7]
            }

task9Dict =  {
                "task_type": "seq_bio",
                "column_idx": 12,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[8]
            }

task10Dict =  {
                "task_type": "seq_bio",
                "column_idx": 13,
                "metric": "span_f1",
                "additional_metrics": ["f1_micro","f1_macro"],
                "loss_weight": weights[9]
            }

            
data = {
	    "train_data_path": pathTrain,
        "validation_data_path": pathDev,
        "word_idx": 3,
        "tasks": {
                "Smell_Word":task1Dict,
                "Smell_Source":task2Dict,
                "Quality":task3Dict,
                "Evoked_Odorant":task4Dict,
                "Location":task5Dict,
                "Perceiver":task6Dict,
                "Circumstances":task7Dict,
                "Odour_Carrier":task8Dict,
                "Effect":task9Dict,
                "Time":task10Dict
        		}

}

final = {
    "smell-fr":data
}

out = json.dumps(final, indent=4)
print(out)