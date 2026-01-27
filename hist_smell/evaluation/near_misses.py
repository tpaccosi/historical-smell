from typing import List, Dict, Tuple, Union, Set
import re
from collections import defaultdict
import numpy as np
from flair.training_utils import Result


###################
# Evaluation code #
###################

def read_pred_tag_file(test_tagged_file: str):
    """Read the tag prediction file for a model, which has two columns:

    1. token
    2. predicted label
    """
    with open(test_tagged_file, 'rt') as fh:
        sent_idx = 0
        token_idx = 0
        for li, line in enumerate(fh):
            parts = line.strip().split(' ')
            if len(parts) == 2:
                token_idx += 1
                yield [sent_idx, token_idx] + parts
            else:
                yield None, None, parts[0], None
                sent_idx += 1
                token_idx = 0
    return None


def read_test_tag_file(test_tagged_file: str):
    """Read the tagged test file for a model, which has three columns:

    1. token
    2. true label
    3. predicted label
    """
    with open(test_tagged_file, 'rt') as fh:
        sent_idx = 0
        token_idx = 0
        for li, line in enumerate(fh):
            parts = line.strip().split(' ')
            if len(parts) == 3:
                token_idx += 1
                yield [sent_idx, token_idx] + parts
            else:
                yield None, None, parts[0], None, None
                sent_idx += 1
                token_idx = 0
    return None


class Token:

    def __init__(self, sent_idx: int, token_idx: int, text: str, label: str):
        self.sent_idx = sent_idx
        self.token_idx = token_idx
        self.text = text
        self.label = label


class Span:

    def __init__(self, sent_idx: int, start, end: int, text, label: Union[str, List[str]]):
        self.sent_idx = sent_idx
        self.start = start
        self.end = end
        self.text = text
        self.label = label

    def __repr__(self):
        return (f"{self.__class__.__name__}(sent_idx={self.sent_idx}, start={self.start}, "
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
        return f'{self.sent_idx}: Span[{self.start}:{self.end}]: "{self.text}"'


def parse_span(span: str, label: str):
    m = re.match(r"(\d+): Span\[(\d+):(\d+)]: \"(.*?)\"", span)
    if m is None:
        raise ValueError(f"invalid span format: {span}")
    sent_idx, start, end, text = m.groups()
    return Span(int(sent_idx), int(start), int(end), text, label)


def make_span(sent_idx, start, end, text):
    return f'{sent_idx}: Span[{start}:{end}]: "{text}"'


def merge_spans(spans: List[Span], label: str = None):
    """Merge a list of Span instances into a single Span. If a label is passed, that is used
    for the merged span, otherwise, the label of the first span is used."""
    merge_texts = []
    merge_start = None
    merge_end = None
    merge_sent = None
    if label is None:
        label = list(set([label for span in spans for label in span.get_labels()]))
    if len(set([span.sent_idx for span in spans])) != 1:
        raise ValueError(f"Not all spans have the same sent_idx: {spans}")
    for span in spans:
        merge_sent = span.sent_idx
        if merge_start is None:
            merge_start = span.start
        merge_end = span.end
        merge_texts.append(span.text)
    # '57: Span[114:121]: "haer Hoogh Mog . Resolutie en bygevoeghde"'
    return Span(merge_sent, merge_start, merge_end, ' '.join(merge_texts), label)


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


def score_strict_lenient(true_spans: List[Span] = None, pred_spans: List[Span] = None,
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


def get_span_from_tokens(tokens: List[Token]) -> Span:
    sent = tokens[0].sent_idx
    start = tokens[0].token_idx
    end = tokens[-1].token_idx + 1
    label = list(set([token.label for token in tokens]))
    text = ' '.join([token.text for token in tokens])
    span = Span(sent, start, end, text, label)
    return span


def get_spans(test_tag_file: str, label_col: str = None) -> List[Span]:
    spans = []
    tokens = []
    read_func = read_test_tag_file if label_col is not None else read_pred_tag_file
    for tag_row in read_func(test_tag_file):
        if label_col is not None:
            sent_idx, token_idx, text, true_label, pred_label = tag_row
            label = true_label if label_col == 'true' else pred_label
        else:
            sent_idx, token_idx, text, label = tag_row
        if label is None:
            tokens = []
            # print('\n', token_idx, token, label, '\n')
            continue
        if label == 'O' or label.startswith('B'):
            if len(tokens) > 0:
                span = get_span_from_tokens(tokens)
                spans.append(span)
                # print()
            tokens = []
        if label.startswith('B') or label.startswith('I'):
            label_type = label[2:]
            if len(tokens) > 0 and tokens[-1].label != label_type:
                span, layer = get_span_from_tokens(tokens)
                spans.append(span)
                # print()
                tokens = []
            tokens.append(Token(sent_idx, token_idx, text, label_type))
        if label != 'O':
            # print(token_idx, token, label, tokens)
            pass
    if len(tokens) > 0:
        span = get_span_from_tokens(tokens)
        spans.append(span)
        # print()
    return spans
