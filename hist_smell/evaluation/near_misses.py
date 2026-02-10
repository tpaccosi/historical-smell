from typing import List, Dict, Tuple, Union, Set
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from flair.training_utils import Result


###################
# Evaluation code #
###################


class Token:

    def __init__(self, text_id: str, sent_idx: int, token_idx: int, text: str, labels: List[str]):
        self.text_id = text_id
        self.sent_idx = sent_idx
        self.token_idx = token_idx
        self.text = text
        self.labels = labels
        self.col = {0: text_id, 1: sent_idx, 2: token_idx, 3: text}
        for col_idx, label in enumerate(labels):
            self.col[col_idx + 4] = label

    def __getitem__(self, key):
        return self.col[key]


def read_pred_file(pred_file: str, sep: str = '\t'):
    """Read the prediction file for a model, which has at least five columns:
    1. text id
    2. sentence index
    3. token index
    4. the text token
    5. one or more predicted labels
    """
    with open(pred_file, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if line == '':
                continue
            try:
                text_id, sent_token, char_range, token_text, *labels = line.split(sep)
            except BaseException:
                print(f"line: #{line}#")
                raise
            sent_idx, token_idx = [int(x) for x in sent_token.split('-')]
            # text_id, sent_idx, token_idx, token_text, *labels = line.strip('\n').split(sep)
            yield Token(text_id, sent_idx, token_idx, token_text, labels)


def write_pred_file(pred_file: str, tokens: List[Token], sep: str = '\t'):
    with open(pred_file, 'wt') as fh:
        for token in tokens:
            label_string = sep.join(token.labels)
            fh.write(f"{token.text_id}{sep}{token.sent_idx}-{token.token_idx}{sep}{token.text}{sep}{label_string}\n")



def read_pred_tag_file(test_tagged_file: str, sep: str = '\t'):
    """Read the tag prediction file for a model, which has two columns:

    1. token
    2. predicted label
    """
    with open(test_tagged_file, 'rt') as fh:
        sent_idx = 0
        token_idx = 0
        for li, line in enumerate(fh):
            parts = line.strip().split(sep)
            if len(parts) == 2:
                token_idx += 1
                yield parts
            else:
                yield None, None, parts[0], None
                sent_idx += 1
                token_idx = 0
    return None


def read_test_tag_file(test_tagged_file: str, sep: str = '\t'):
    """Read the tagged test file for a model, which has six columns:

    1. text_id
    2. sentence_index
    3. token_index
    4. token
    5. true label
    6. predicted label
    """
    with open(test_tagged_file, 'rt') as fh:
        sent_idx = 0
        token_idx = 0
        for li, line in enumerate(fh):
            parts = line.strip().split(sep)
            if len(parts) == 6:
                text_id, sent_idx, token_idx, token, true_label, pred_label = parts
                yield text_id, int(sent_idx), int(token_idx), token, true_label, pred_label
            else:
                yield None, None, parts[0], None, None
                sent_idx += 1
                token_idx = 0
    return None


class Span:

    def __init__(self, text_id: str, sent_idx: int, start, end: int, text, label: Union[str, List[str]]):
        self.text_id = text_id
        self.sent_idx = sent_idx
        self.start = start
        self.end = end
        self.text = text
        self.label = label

    def __repr__(self):
        return (f"{self.__class__.__name__}(text_id={self.text_id}, sent_idx={self.sent_idx}, start={self.start}, "
                f"end={self.end}, text={self.text}, label={self.label}")

    def has_label(self, label: str):
        return self.label == label if isinstance(self.label, str) else label in self.label

    def get_labels(self):
        if isinstance(self.label, str):
            return [self.label]
        else:
            return list(self.label)

    @property
    def string(self):
        return f'{self.text_id} {self.sent_idx}: Span[{self.start}:{self.end}]: "{self.text}"'


def parse_span(span: str, label: str):
    m = re.match(r"(\S+) (\d+): Span\[(\d+):(\d+)]: \"(.*?)\"", span)
    if m is None:
        raise ValueError(f"invalid span format: {span}")
    text_id, sent_idx, start, end, text = m.groups()
    return Span(text_id, int(sent_idx), int(start), int(end), text, label)


def make_span(text_id, sent_idx, start, end, text):
    return f'{text_id} {sent_idx}: Span[{start}:{end}]: "{text}"'


def merge_spans(spans: List[Span], label: str = None):
    """Merge a list of Span instances into a single Span. If a label is passed, that is used
    for the merged span, otherwise, the label of the first span is used."""
    merge_texts = []
    merge_start = None
    merge_end = None
    merge_sent = None
    merge_text_id = None
    if label is None:
        label = list(set([label for span in spans for label in span.get_labels()]))
    if len(set([span.text_id for span in spans])) != 1:
        raise ValueError(f"Not all spans have the same text_id: {spans}")
    if len(set([span.sent_idx for span in spans])) != 1:
        raise ValueError(f"Not all spans have the same sent_idx: {spans}")
    for span in spans:
        merge_text_id = span.text_id
        merge_sent = span.sent_idx
        if merge_start is None:
            merge_start = span.start
        merge_end = span.end
        merge_texts.append(span.text)
    # '57: Span[114:121]: "haer Hoogh Mog . Resolutie en bygevoeghde"'
    return Span(merge_text_id, merge_sent, merge_start, merge_end, ' '.join(merge_texts), label)


def find_overlapping_spans(test_spans: List[Span], pred_spans: List[Span]) -> List[Tuple[Span, Span]]:
    """Find overlapping spans between two lists of spans. Returns a list of tuples of overlapping spans."""
    overlapping_spans = []
    sent_test_spans = defaultdict(list)
    sents = set([(span.text_id, span.sent_idx) for span in test_spans])
    sents.update([(span.text_id, span.sent_idx) for span in pred_spans])
    sents = sorted(sents)
    for span in test_spans:
        sent_test_spans[(span.text_id, span.sent_idx)].append(span)
    sent_pred_spans = defaultdict(list)
    for span in pred_spans:
        sent_pred_spans[(span.text_id, span.sent_idx)].append(span)
    for sent in sents:
        matched = set()
        for test_span in sent_test_spans[sent]:
            for pred_span in sent_pred_spans[sent]:
                if test_span.text_id != pred_span.text_id:
                    continue
                if test_span.sent_idx != pred_span.sent_idx:
                    continue
                if test_span.label != pred_span.label:
                    continue
                if spans_match(test_span, pred_span):
                    overlapping_spans.append((test_span, pred_span))
                    matched.update([test_span, pred_span])
                elif max(test_span.start, pred_span.start) <= min(test_span.end, pred_span.end):
                    overlapping_spans.append((test_span, pred_span))
                    matched.update([test_span, pred_span])
            if test_span not in matched:
                overlapping_spans.append((test_span, None))
        for pred_span in sent_pred_spans[sent]:
            if pred_span not in matched:
                overlapping_spans.append((None, pred_span))
    return overlapping_spans


def classify_start_overlap(row):
    if row['match_type'].startswith('miss_pred'):
        return 'missed'
    if row['match_type'].startswith('miss_test'):
        return 'wrong'
    if row['test_start'] == row['pred_start']:
        return 'exact'
    if row['test_start'] < row['pred_start']:
        return 'late'
    else:
        return 'early'


def classify_end_overlap(row):
    if row['match_type'].startswith('miss_pred'):
        return 'missed'
    if row['match_type'].startswith('miss_test'):
        return 'wrong'
    if row['test_end'] == row['pred_end']:
        return 'exact'
    if row['test_end'] < row['pred_end']:
        return 'late'
    else:
        return 'early'


def make_overlapping_spans_dataframe(overlapping_spans: List[Tuple[Span, Span]]):
    """Convert a list of overlapping spans into a pandas DataFrame."""
    rows = []
    for test_span, pred_span in overlapping_spans:
        if pred_span is None:
            match_type = 'miss_pred'
        elif test_span is None:
            match_type = 'miss_test'
        elif spans_match(test_span, pred_span):
            match_type = 'exact'
        else:
            match_type = 'partial'
        row = {
            'test_text_id': test_span.text_id if test_span is not None else None,
            'test_sent_idx': test_span.sent_idx if test_span is not None else None,
            'test_start': test_span.start if test_span is not None else None,
            'test_end': test_span.end if test_span is not None else None,
            'test_text': test_span.text if test_span is not None else None,
            'test_label': test_span.label if test_span is not None else None,
            'pred_text_id': pred_span.text_id if pred_span is not None else None,
            'pred_sent_idx': pred_span.sent_idx if pred_span is not None else None,
            'pred_start': pred_span.start if pred_span is not None else None,
            'pred_end': pred_span.end if pred_span is not None else None,
            'pred_text': pred_span.text if pred_span is not None else None,
            'pred_label': pred_span.label if pred_span is not None else None,
            'match_type': match_type
        }
        row['overlap_start'] = classify_start_overlap(row)
        row['overlap_end'] = classify_end_overlap(row)
        rows.append(row)
    return pd.DataFrame(rows)


def get_extended_res_spans(pred_spans: List[Span]) -> List[Span]:
    extended_res_spans = []
    for si, curr_span in enumerate(pred_spans):
        if 'RES' not in curr_span.label:
            continue
        res_spans = [curr_span]
        curr_end = curr_span.end
        next_idx = si + 1
        while curr_end is not None:
            if next_idx >= len(pred_spans):
                break
            next_span = pred_spans[next_idx]
            if next_span.sent_idx != curr_span.sent_idx:
                curr_end = None
            elif next_span.label == 'RES':
                curr_end = None
            elif next_span.start != curr_end:
                curr_end = None
            else:
                res_spans.append(next_span)
                curr_end = next_span.end
            next_idx += 1
        merged_span = merge_spans(res_spans)
        extended_res_spans.append(merged_span)
    return extended_res_spans


def have_same_sent(span1: Span, span2: Span) -> bool:
    return span1.sent_idx == span2.sent_idx


def filter_partial_matches(pred_only_spans: Union[Set[Span], List[Span]],
                           true_only_spans: Union[Set[Span], List[Span]]):
    matched = defaultdict(list)
    for pred_only_span in sorted(pred_only_spans, key=lambda s: (s.sent_idx, s.start)):
        if pred_only_span in matched:
            print('pred_only_span already in matched:', pred_only_span)
            pass
        for true_only_span in true_only_spans:
            if have_same_sent(pred_only_span, true_only_span) is False:
                continue
            if pred_only_span.end < true_only_span.start or true_only_span.end < pred_only_span.start:
                continue
            if pred_only_span in matched:
                pass
            matched[pred_only_span].append(true_only_span)
    return matched


def get_true_pred_spans_from_results(results: Result, target_label: str) -> Tuple[List[Span], List[Span]]:
    # extract and parse tagged spans
    true_spans = [parse_span(span, label) for span, label in results.all_true_values.items()]
    pred_spans = [parse_span(span, label) for span, label in results.all_predicted_values.items()]
    # filter by label
    true_spans = [span for span in true_spans if span.has_label(target_label)]
    if target_label == 'RES':
        pred_spans = get_extended_res_spans(pred_spans)
    else:
        pred_spans = [span for span in pred_spans if span.has_label(target_label)]
    return true_spans, pred_spans


def get_matching_spans(true_spans: List[Span], pred_spans: List[Span]) -> Tuple[List[Span], List[Span], List[Span]]:
    true_string_span = {span.string: span for span in true_spans}
    pred_string_spans = {span.string: span for span in pred_spans}
    pred_true_spans = []
    pred_only_spans = []
    true_only_spans = []
    for span in pred_spans:
        if span.string in true_string_span:
            pred_true_spans.append(span)
        else:
            pred_only_spans.append(span)
    for span in true_spans:
        if span.string not in pred_string_spans:
            true_only_spans.append(span)
    return pred_true_spans, true_only_spans, pred_only_spans


def spans_match(span1: Span, span2: Span) -> bool:
    if span1.string != span2.string:
        return False
    labels1 = set(span1.label)
    labels2 = set(span2.label)
    return labels1 == labels2


def compute_f_score(row, eval_type, f: float = 1.0):
    prec_col = f"{eval_type}_prec"
    rec_col = f"{eval_type}_rec"
    if row[prec_col] + row[rec_col] == 0.0:
        return 0.0
    else:
        return 2 * row[prec_col] * row[rec_col] / (row[prec_col] + row[rec_col])


def make_match_frame(overlap_df, test_freq, pred_freq):
    exact_label_freq = overlap_df[overlap_df.match_type == 'exact'].test_label.value_counts().rename('exact_freq')

    partial_label_freq = overlap_df[overlap_df.match_type == 'partial'].test_label.value_counts().rename('partial_freq')

    match_df = pd.concat([test_freq, pred_freq, exact_label_freq, partial_label_freq], axis=1).fillna(0.0)
    match_df['missing'] = match_df.apply(lambda row: row['test_freq'] - row['exact_freq'] - row['partial_freq'], axis=1)
    match_df['lenient_freq'] = match_df.exact_freq + match_df.partial_freq
    match_df['exact_prop'] = match_df.exact_freq / match_df.exact_freq.sum()
    match_df['partial_prop'] = match_df.partial_freq / match_df.partial_freq.sum()
    match_df['lenient_prop'] = match_df.lenient_freq / match_df.lenient_freq.sum()

    match_df.loc['Overall'] = match_df.sum()
    match_df['exact_frac'] = match_df.exact_freq / match_df.test_freq
    match_df['partial_frac'] = match_df.partial_freq / match_df.test_freq
    match_df['exact_prec'] = match_df.exact_freq / match_df.pred_freq
    match_df['exact_rec'] = match_df.exact_freq / match_df.test_freq
    match_df['exact_f1'] = match_df.apply(lambda row: compute_f_score(row, 'exact', f=1.0), axis=1)
    match_df['lenient_prec'] = match_df.lenient_freq / match_df.pred_freq
    match_df['lenient_rec'] = match_df.lenient_freq / match_df.test_freq
    match_df['lenient_f1'] = match_df.apply(lambda row: compute_f_score(row, 'lenient', f=1.0), axis=1)
    return match_df


def score_strict_lenient(overlap_df, test_freq, pred_freq):
    match_df = make_match_frame(overlap_df, test_freq, pred_freq)
    weighted_avg_exact = (match_df[['exact_prec', 'exact_rec', 'exact_f1']].T.drop('Overall', axis=1) * match_df.drop(
        'Overall').exact_prop).T.sum()
    weighted_avg_lenient = (
                match_df[['lenient_prec', 'lenient_rec', 'lenient_f1']].T.drop('Overall', axis=1) * match_df.drop(
            'Overall').lenient_prop).T.sum()
    weighted_avg = pd.concat([weighted_avg_exact, weighted_avg_lenient]).rename('weight_avg')
    eval_cols = ['exact_prec', 'exact_rec', 'exact_f1', 'lenient_prec', 'lenient_rec', 'lenient_f1']
    macro_avg = match_df.drop('Overall')[eval_cols].mean().rename('macro_avg')
    eval_avg = pd.concat([macro_avg, weighted_avg], axis=1).T
    score_df = pd.concat([match_df[eval_cols], eval_avg])
    return score_df


def score_strict_lenient_old(true_spans: List[Span] = None, pred_spans: List[Span] = None,
                         result: Result = None, label: str = None) -> Dict[str, any]:
    if result is not None:
        true_spans, pred_spans = get_true_pred_spans_from_results(result, label)
    pred_true_spans, true_only_spans, pred_only_spans = get_matching_spans(true_spans, pred_spans)
    if label is None:
        label = true_spans[0].label

    partial_matches = filter_partial_matches(pred_only_spans, true_only_spans)
    pred_partial = len(partial_matches)
    true_partial = sum([len(true_parts) for _, true_parts in partial_matches.items()])

    # print(f"true_spans: {len(true_spans)}\t"
    #       f"pred_only_spans: {len(pred_only_spans)}\t"
    #       f"true_only_spans: {len(true_only_spans)}")
    # print(f"exact_matches: {len(pred_true_spans)}\tpartial_matches: {len(partial_matches)}")
    if len(true_spans) == 0:
        strict_rec = np.nan
        lenient_rec = np.nan
    else:
        strict_rec = len(pred_true_spans) / len(true_spans)
        lenient_rec = (len(pred_true_spans) + true_partial) / len(true_spans)

    if len(pred_spans) == 0:
        if len(true_spans) == 0:
            strict_prec = np.nan
            lenient_prec = np.nan
        else:
            strict_prec = 0.0
            lenient_prec = 0.0
    else:
        strict_prec = len(pred_true_spans) / len(pred_spans)
        lenient_prec = (len(pred_true_spans) + pred_partial) / len(pred_spans)

    if len(true_spans) == 0:
        strict_f1 = np.nan
        lenient_f1 = np.nan
    else:
        strict_f1 = 0.0 if len(pred_true_spans) == 0 else 2 * strict_prec * strict_rec / (strict_prec + strict_rec)
        lenient_f1 = 0.0 if (lenient_prec + lenient_rec) == 0.0 else 2 * lenient_prec * lenient_rec / (
                lenient_prec + lenient_rec)
    # else:
    #     strict_f1 = 2 * strict_prec * strict_rec / (strict_prec + strict_rec)
    #     lenient_f1 = 2 * lenient_prec * lenient_rec / (lenient_prec + lenient_rec)

    scores = {
        'label': label,
        'support': len(true_spans),
        'true_pred': len(pred_true_spans),
        'true_only': len(true_only_spans),
        'pred_only': len(pred_only_spans),
        'true_partial': true_partial,
        'pred_partial': pred_partial,
        'precision_strict': strict_prec,
        'precision_lenient': lenient_prec,
        'recall_strict': strict_rec,
        'recall_lenient': lenient_rec,
        'f1_strict': strict_f1,
        'f1_lenient': lenient_f1,
    }

    # print(f"prec. strict: {strict_prec: >6.3f}\tlenient: {lenient_prec: >6.3f}")
    # print(f"recall strict: {strict_rec: >6.3f}\tlenient: {lenient_rec: >6.3f}")
    # print(f"F-1 strict: {strict_f1: >6.3f}\tlenient: {lenient_f1: >6.3f}")
    return scores


def get_span_from_tokens(tokens: List[Token], label_col: int) -> Span:
    text_id = tokens[0].text_id
    sent = tokens[0].sent_idx
    start = tokens[0].token_idx
    end = tokens[-1].token_idx + 1
    label = tokens[0][label_col][2:]
    text = ' '.join([token.text for token in tokens])
    span = Span(text_id, sent, start, end, text, label)
    return span


def tokens_to_spans(doc_tokens: List[Token]) -> List[Span]:
    spans = []
    tokens = []
    if len(doc_tokens) == 0:
        return spans
    for label_col in range(5, len(doc_tokens[0].col)):
        prev_text_id = None
        prev_sent_idx = None
        for token in doc_tokens:
            label = token[label_col]
            # print(f'{token.token_idx:>4d} {token.text:<10s} {label:<10s}')
            if label is None or prev_text_id != token.text_id or prev_sent_idx != token.sent_idx:
                tokens = []
                prev_text_id, prev_sent_idx = None, None
                # print(f'\nsent boundary\n')
                if label is None:
                    continue
            if label == 'O' or label.startswith('B'):
                # print(f'\nspan boundary\n')
                if len(tokens) > 0:
                    span = get_span_from_tokens(tokens, label_col=label_col)
                    spans.append(span)
                    # print()
                tokens = []
            if label.startswith('B') or label.startswith('I'):
                label_type = label[2:]
                if len(tokens) > 0 and tokens[-1][label_col][2:] != label_type:
                    span = get_span_from_tokens(tokens, label_col=label_col)
                    spans.append(span)
                    # print()
                    tokens = []
                tokens.append(token)
            if label != 'O':
                # print(token_idx, token, label, tokens)
                pass
            prev_text_id, prev_sent_idx = token.text_id, token.sent_idx
            # print(f"number of spans: {len(spans)}\tnumber of tokens: {len(tokens)}")
        if len(tokens) > 0:
            span = get_span_from_tokens(tokens, label_col=label_col)
            spans.append(span)
            # print()
    return spans


def get_spans(test_tag_file: str, label_col: str = None) -> List[Span]:
    spans = []
    tokens = []
    read_func = read_test_tag_file if label_col is not None else read_pred_tag_file
    for tag_row in read_func(test_tag_file):
        if label_col is not None:
            text_id, sent_idx, token_idx, text, true_label, pred_label = tag_row
            label = true_label if label_col == 'true' else pred_label
        else:
            text_id, sent_idx, token_idx, text, label = tag_row
    return spans
