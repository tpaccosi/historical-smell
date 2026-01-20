import re
import sys
import gzip

from bs4 import BeautifulSoup
from collections import Counter


def extract_dutch_prose_text(soup):
    dutch_prose_text = extract_dutch_prose_text_strings(soup)
    return '\n'.join(dutch_prose_text)


def filter_tokens(doc, skip_first, skip_last, debug: int = 0):
    num_tokens = len(doc)
    cumu_tokens = 0
    selected_sents = []
    for si, sent in enumerate(doc.sents):
        if cumu_tokens < skip_first:
            if debug > 0:
                print(f'skip_first skipping sent {si}', num_tokens, cumu_tokens)
            pass
        elif num_tokens - cumu_tokens < skip_last:
            if debug > 0:
                print(f'skip_last skipping sent {si}', num_tokens, cumu_tokens)
            pass
        else:
            selected_sents.append(sent)
        cumu_tokens += len(sent)
    return selected_sents


def chunk_doc(doc, chunk_size: int = 5000, skip_first: int = 0, skip_last: int = 0):
    chunks = []
    chunk = []
    selected_sents = filter_tokens(doc, skip_first, skip_last)
    for si, sent in enumerate(selected_sents):
        chunk.extend([token for token in sent if token.pos_ not in {'SPACE', 'PUNCT'}])
        if len(chunk) > chunk_size:
            chunks.append(chunk)
            chunk = []
    if len(chunk) > 0:
        chunks.append(chunk)
    return chunks


def make_chunk_rows(ti_id, chunks):
    rows = []
    for ci, chunk in enumerate(chunks):
        chunk_id = f"{ti_id}-chunk-{ci + 1:0>3}"
        term_freq = Counter([(token.lemma_, token.pos_) for token in chunk])
        for term, pos in term_freq:
            row = [ti_id, chunk_id, term, pos, term_freq[(term, pos)]]
            rows.append(row)
    return rows


def is_div(soup):
    return soup.name == 'div'


def is_non_prose_div(soup):
    if not is_div(soup):
        return False
    if 'ebook-type' in soup.attrs:
        return True
    elif 'wpg-type' in soup.attrs:
        return True
    else:
        return False


def is_prose_div(soup):
    if not is_div(soup):
        return False
    return not is_non_prose_div(soup)


def is_non_dutch_element(soup):
    if 'lang' not in soup.attrs:
        return False
    return soup.attrs['lang'] != 'nl'


def extract_non_prose_divs(soup):
    divs = soup.find_all('div')
    return [div for div in divs if is_non_prose_div(div)]


def extract_prose_divs(soup):
    divs = [div for div in soup.find_all('div')]
    return [div for div in divs if is_prose_div(div)]


def extract_non_prose_paragraphs(soup):
    non_prose_divs = extract_non_prose_divs(soup)
    return set([para for div in non_prose_divs for para in div.find_all('p')])


def extract_prose_paragraphs(soup):
    paras = soup.find_all('p')
    non_prose_paras = extract_non_prose_paragraphs(soup)
    return set([para for para in paras if para not in non_prose_paras])


def extract_non_dutch_divs(soup):
    divs = [div for div in soup.find_all('div')]
    return [div for div in divs if is_non_dutch_element(div)]


def extract_non_dutch_paragraphs(soup):
    """Return all paragraphs that are marked non-Dutch or that are in
    a div that is marked as non-dutch."""
    prose_divs = extract_prose_divs(soup)

    # paras of divs marked as non-dutch
    non_dutch_divs = extract_non_dutch_divs(soup)
    non_dutch_div_paras = [para for div in non_dutch_divs for para in div.find_all('p')]

    # prose paras marked as non-dutch
    prose_paras = extract_prose_paragraphs(soup)
    non_dutch_prose_paras = [para for para in prose_paras if is_non_dutch_element(para)]

    return set(non_dutch_div_paras + non_dutch_prose_paras)


def extract_dutch_paragraphs(soup):
    paras = soup.find_all('p')
    non_dutch_paras = extract_non_dutch_paragraphs(soup)
    return set([para for para in paras if para not in non_dutch_paras])


def extract_dutch_prose_text_strings(soup):
    paras = soup.find_all('p')
    # print(f"all paras: {len(paras)}")
    dutch_paras = extract_dutch_paragraphs(soup)
    # print(f"dutch paras: {len(dutch_paras)}")
    prose_paras = extract_prose_paragraphs(soup)
    # print(f"prose paras: {len(prose_paras)}")
    dutch_prose_paras = set([para for para in dutch_paras if para in prose_paras])
    # print(f"dutch prose paras: {len(dutch_prose_paras)}")
    # make sure the paras are in document order
    ordered_paras = []
    for para in paras:
        if para not in dutch_prose_paras:
            continue
        if para in ordered_paras:
            continue
        ordered_paras.append(para)

    # print(f"ordered dutch prose paras: {len(ordered_paras)}")
    return [text for para in ordered_paras for text in para.stripped_strings]

