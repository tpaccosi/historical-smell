import pandas as pd


def map_predictions_to_bio_format(predictions, tokenized_test, ids_to_labels):
    """
    Maps predictions to BIO format for evaluation.

    :param predictions: Model predictions
    :param tokenized_test: Tokenized test dataset
    :param ids_to_labels: Mapping from label IDs to label names
    :return: List of rows with prediction details
    """
    rows = []
    for idx, batch in enumerate(tokenized_test):
        batch_preds = predictions[idx][:len(batch['labels'])]
        prev_id = None
        for pred, label, word_id in zip(batch_preds, batch['labels'], batch['word_ids']):
            if word_id is None:
                continue
            if word_id == prev_id:
                continue
            row = {
                'text_id': batch['Document'],
                'sent_idx': batch['Num'],
                'token_idx': word_id + 1,
                'token': batch['sentence'][word_id],
                'label': ids_to_labels[label],
                'pred': ids_to_labels[pred],
            }
            rows.append(row)
            prev_id = word_id
    return pd.DataFrame(rows)
