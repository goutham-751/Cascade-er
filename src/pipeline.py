"""Local, bounded-memory business entity resolution baseline. See README.md."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import resource
import sqlite3
import sys
import time
import unicodedata
from functools import lru_cache
import regex
from anyascii import anyascii
from retrieval_extras import extra_keys, phonetic, VERSION as EXTRA_VERSION

import numpy as np
from rapidfuzz import fuzz, process

FIELDS = ['entity_id', 'business_name', 'business_address', 'country']
INDEX_VERSION = 2
RETRIEVAL_VERSION = 7
WORDS = regex.compile(r'[\p{L}\p{M}\p{N}]+')
LEGAL = set('inc incorporated corp corporation llc ltd limited pvt private co company'.split())
ABBREVIATIONS = dict(street='st', road='rd', avenue='ave', boulevard='blvd',
                     drive='dr', lane='ln', apartment='apt', suite='ste')
ROMAN_LEGAL = LEGAL | {'praivet', 'preivet', 'praivett', 'pra', 'li', 'llp', 'pllc'}


@lru_cache(maxsize=100000)
def roman_core(text):
    value = anyascii(text).lower().replace("'", '')
    return ' '.join(t for t in re.findall('[a-z0-9]+', value) if t not in ROMAN_LEGAL)


@lru_cache(maxsize=100000)
def numeric_tokens(text):
    return frozenset(str(int(n)) for n in re.findall(r'\b[0-9]+\b', text))


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        yield from csv.DictReader(f, delimiter='\t')


def write_rows(path, records, fields=FIELDS):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(records)


def bucket(value, modulo=10000):
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), 'big') % modulo


def normalize(value):
    # Strip Latin accents, preserving vowel signs in Indian scripts.
    text = re.sub('[\u0300-\u036f]', '', unicodedata.normalize('NFKD', value.casefold()))
    return ' '.join(WORDS.findall(text.replace('&', ' and ')))


def record(row):
    name = normalize(row['business_name'])
    core = ' '.join(t for t in name.split() if t not in LEGAL) or name
    address = ' '.join(ABBREVIATIONS.get(t, t) for t in normalize(row['business_address']).split())
    return (row['entity_id'], name, core, address, normalize(row['country']))


def keys(rec):
    from retrieval_extras import phonetic
    _, name, core, address, country = rec
    # Include 2-char tokens (important for Asian businesses) and up to 8 tokens
    tokens = sorted(set(t for t in core.split() if len(t) >= 2), key=lambda t: (-len(t), t))[:8]
    result = {country + '|t|' + t[:5] for t in tokens}
    
    # Add phonetic blocking keys for the longest 3 tokens to catch severe spelling mistakes
    for t in tokens[:3]:
        ph = phonetic(t)
        if ph:
            result.add(country + '|ph|' + ph)

    compact = core.replace(' ', '')
    if compact:
        result.add(country + '|n|' + compact[:8])
        # Cross-border compact name match (in case country label is wrong/missing)
        result.add('xb|n|' + compact[:8])
        
    numbers = re.findall(r'\b\d+\b', address)
    if numbers and compact:
        result.add(country + '|h|' + numbers[0] + '|' + compact[:3])
    for number in numbers:
        if len(number) in (5, 6) and compact:
            result.add(country + '|p|' + number + '|' + compact[:2])
            
    if core:
        core_hash = hashlib.blake2b(' '.join(sorted(core.split())).encode(), digest_size=8).hexdigest()
        result.add(country + '|e|' + core_hash)
        # Cross-border exact match
        result.add('xb|e|' + core_hash)
        
    if address:
        result.add(country + '|a|' + hashlib.blake2b(address.encode(), digest_size=8).hexdigest())
        
    # Name-independent retrieval reaches transliterated and renamed businesses.
    addr_tokens = sorted({t for t in address.split() if len(t) >= 3 and not t.isdigit()
                          and t not in {'null', 'floor', 'near', 'india', 'opposite', 'building', 'street', 'road', 'avenue'}},
                         key=lambda t: (-len(t), t))[:6]
    for number in list(dict.fromkeys(numbers))[:4]:
        for token in addr_tokens:
            result.add(country + '|at|' + number + '|' + token[:6])
    return sorted(result)


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 0.0


FEATURE_NAMES = ['name_ratio', 'name_sorted', 'name_set', 'core_ratio', 'core_sorted',
                 'address_ratio', 'address_sorted', 'address_set', 'name_jaccard',
                 'address_jaccard', 'number_jaccard', 'first_number_equal',
                 'postal_overlap', 'name_exact', 'address_exact', 'name_length_ratio',
                 'address_length_ratio', 'name_missing', 'address_missing']
FEATURE_NAMES += ['roman_name_ratio', 'roman_name_sorted', 'roman_name_set',
                  'normalized_number_jaccard', 'normalized_number_containment', 'different_script']
FEATURE_NAMES += ['compact_name_ratio', 'compact_name_partial', 'phonetic_name_ratio',
                  'name_initials_ratio', 'address_letters_sorted', 'address_letters_set',
                  'fuzzy_number_best', 'fuzzy_number_mean', 'name_token_containment',
                  'address_token_containment', 'anchor_name_frequency', 'target_name_frequency']

def features(a, b, con=None):
    _, an, ac, aa, _ = a
    _, bn, bc, ba, _ = b
    nums_a, nums_b = re.findall(r'\b\d+\b', aa), re.findall(r'\b\d+\b', ba)
    post_a, post_b = {n for n in nums_a if len(n) in (5, 6)}, {n for n in nums_b if len(n) in (5, 6)}
    return [fuzz.ratio(an, bn)/100, fuzz.token_sort_ratio(an, bn)/100,
            fuzz.token_set_ratio(an, bn)/100, fuzz.ratio(ac, bc)/100,
            fuzz.token_sort_ratio(ac, bc)/100, fuzz.ratio(aa, ba)/100,
            fuzz.token_sort_ratio(aa, ba)/100, fuzz.token_set_ratio(aa, ba)/100,
            overlap(ac.split(), bc.split()), overlap(aa.split(), ba.split()),
            overlap(nums_a, nums_b), float(bool(nums_a and nums_b) and nums_a[0] == nums_b[0]),
            float(bool(post_a & post_b)), float(bool(an) and an == bn),
            float(bool(aa) and aa == ba), min(len(ac), len(bc))/max(len(ac), len(bc), 1),
            min(len(aa), len(ba))/max(len(aa), len(ba), 1),
            float(not an or not bn), float(not aa or not ba),
            fuzz.ratio(roman_core(ac), roman_core(bc))/100,
            fuzz.token_sort_ratio(roman_core(ac), roman_core(bc))/100,
            fuzz.token_set_ratio(roman_core(ac), roman_core(bc))/100,
            overlap(numeric_tokens(aa), numeric_tokens(ba)),
            len(numeric_tokens(aa) & numeric_tokens(ba))/max(1, min(len(numeric_tokens(aa)), len(numeric_tokens(ba)))),
            float(ac.isascii() != bc.isascii()),
            fuzz.ratio(roman_core(ac).replace(' ', ''), roman_core(bc).replace(' ', ''))/100,
            fuzz.partial_ratio(roman_core(ac).replace(' ', ''), roman_core(bc).replace(' ', ''))/100,
            fuzz.ratio(phonetic(ac), phonetic(bc))/100,
            max(fuzz.ratio(''.join(t[0] for t in ac.split()),bc.replace(' ','')),
                fuzz.ratio(''.join(t[0] for t in bc.split()),ac.replace(' ','')))/100,
            fuzz.token_sort_ratio(re.sub('[0-9]+','',aa),re.sub('[0-9]+','',ba))/100,
            fuzz.token_set_ratio(re.sub('[0-9]+','',aa),re.sub('[0-9]+','',ba))/100,
            number_similarity(aa, ba)[0], number_similarity(aa, ba)[1],
            len(set(ac.split())&set(bc.split()))/max(1,min(len(set(ac.split())),len(set(bc.split())))),
            len(set(aa.split())&set(ba.split()))/max(1,min(len(set(aa.split())),len(set(ba.split())))),
            np.log1p(con.name_frequency(a)) if con else 0.0,
            np.log1p(con.name_frequency(b)) if con else 0.0]


@lru_cache(maxsize=100000)
def number_similarity(a, b):
    na, nb = re.findall('[0-9]+',a), re.findall('[0-9]+',b)
    if not na or not nb:
        return 0.0, 0.0
    best = [max(fuzz.ratio(str(int(x)),str(int(y))) for y in nb)/100 for x in na]
    return max(best), sum(best)/len(best)


class IndexConnection(sqlite3.Connection):
    big_blocks = None
    extra_blocks = None

    @lru_cache(maxsize=100000)
    def extra_frequency(self, key):
        if self.extra_blocks is None:
            return 1
        if key in self.extra_blocks:
            return self.extra_blocks[key]
        return self.execute('SELECT COUNT(*) FROM extra.blocks WHERE key=?',(key,)).fetchone()[0]

    def name_frequency(self, rec):
        core, country = rec[2], rec[4]
        if not core:
            return 0
        key = country+'|e|'+hashlib.blake2b(' '.join(sorted(core.split())).encode(),digest_size=8).hexdigest()
        return self.key_frequency(key)

    @lru_cache(maxsize=100000)
    def key_frequency(self, key):
        if self.big_blocks is None:
            self.big_blocks = dict(self.execute('SELECT key,n FROM large_blocks'))
        if key in self.big_blocks:
            return self.big_blocks[key]
        return self.execute('SELECT COUNT(*) FROM blocks WHERE key=?',(key,)).fetchone()[0]

    @lru_cache(maxsize=10000)
    def lookup(self, key, cap):
        if self.big_blocks is None:
            self.big_blocks = dict(self.execute('SELECT key,n FROM large_blocks'))
        if self.big_blocks.get(key, 0) > cap:
            return ()
        found = self.execute('SELECT rid FROM blocks WHERE key=? LIMIT ?', (key, cap+1)).fetchall()
        return tuple(r[0] for r in found) if len(found) <= cap else ()


def connect(path):
    con = sqlite3.connect(path, factory=IndexConnection)
    con.execute('PRAGMA cache_size=-65536')
    con.execute('PRAGMA temp_store=FILE')
    con.execute('PRAGMA mmap_size=17179869184')
    extra = Path(str(path)+'.extra.sqlite')
    if extra.exists():
        tables = sqlite3.connect(extra)
        ready = tables.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='metadata'").fetchone()[0]
        metadata = tables.execute('SELECT version,records FROM metadata').fetchone() if ready else None
        tables.close()
        if metadata:
            assert metadata == (EXTRA_VERSION, con.execute('SELECT records FROM metadata').fetchone()[0])
            con.execute('ATTACH DATABASE ? AS extra', (str(extra),))
            con.execute('PRAGMA extra.mmap_size=17179869184')
            con.extra_blocks = dict(con.execute('SELECT key,n FROM extra.large_blocks'))
    return con


def build_index(paths, db):
    if Path(db).exists():
        raise FileExistsError(f'{db} already exists; use a new path or remove the old generated index')
    con = connect(db)
    con.executescript('''CREATE TABLE records (rid INTEGER PRIMARY KEY, eid TEXT, name TEXT,
                         core TEXT, address TEXT, country TEXT);
                         CREATE TABLE blocks (key TEXT, rid INTEGER);''')
    n = 0
    rec_batch, key_batch = [], []

    def flush():
        con.executemany('INSERT INTO records VALUES (?,?,?,?,?,?)', rec_batch)
        con.executemany('INSERT INTO blocks VALUES (?,?)', key_batch)
        con.commit()
        rec_batch.clear()
        key_batch.clear()

    for path in paths:
        for row in rows(path):
            n += 1
            rec = record(row)
            rec_batch.append((n, *rec))
            key_batch.extend((key, n) for key in keys(rec))
            if n % 10000 == 0:
                flush()
            if n % 100000 == 0:
                log(f'Indexed {n:,} target records')
    flush()
    log('Building disk lookup index')
    con.execute('CREATE INDEX blocks_key ON blocks(key, rid)')
    log('Counting common blocking keys')
    con.execute('CREATE TABLE large_blocks AS SELECT key,COUNT(*) AS n FROM blocks GROUP BY key HAVING COUNT(*) > 100')
    con.execute('CREATE TABLE metadata (complete INTEGER, records INTEGER, retrieval_version INTEGER)')
    con.execute('INSERT INTO metadata VALUES (1, ?, ?)', (n, INDEX_VERSION))
    con.commit()
    con.close()
    log(f'Index complete: {n:,} records; {Path(db).stat().st_size / 2**20:.1f} MiB')


def candidates(con, rec, top_k=80, max_block=1000):
    if con.big_blocks is None:
        con.big_blocks = dict(con.execute('SELECT key,n FROM large_blocks'))
    all_keys = keys(rec)
    allowed = [key for key in all_keys if con.big_blocks.get(key, 0) <= max_block]
    if max_block < 100:
        allowed = [key for key in allowed if con.lookup(key, max_block)]
    queries, params = [], []
    if allowed:
        queries.append('SELECT rid FROM blocks WHERE key IN ('+','.join('?' for _ in allowed)+')')
        params.extend(allowed)
    # Broad keys can still be selective together. Use at most two bounded joins.
    broad = sorted((key for key in all_keys if max_block < con.big_blocks.get(key, 0) <= 50000),
                   key=lambda key: (con.big_blocks[key], key))
    name_keys = [key for key in broad if key.split('|')[1] in {'t', 'e', 'n'}]
    addr_keys = [key for key in broad if key.split('|')[1] == 'at']
    intersections = []
    if name_keys and addr_keys:
        intersections.append((name_keys[0], addr_keys[0]))
    if len(addr_keys) > 1:
        first = addr_keys[0]
        second = next((key for key in addr_keys[1:] if key.split('|')[2] != first.split('|')[2]), None)
        if second:
            intersections.append((first, second))
    if not intersections:
        tokens = [key for key in name_keys if '|t|' in key]
        if len(tokens) > 1:
            intersections.append((tokens[0], tokens[1]))
    extra_ids = set()
    for first, second in intersections[:2]:
        if con.big_blocks[first] > con.big_blocks[second]:
            first, second = second, first
        found = con.execute('SELECT a.rid FROM blocks a JOIN blocks b ON a.rid=b.rid '
                            'WHERE a.key=? AND b.key=? LIMIT ?', (first, second, max_block+1)).fetchall()
        if len(found) <= max_block:
            extra_ids.update(r[0] for r in found)
    if extra_ids:
        queries.append('SELECT rid FROM records WHERE rid IN ('+','.join('?' for _ in extra_ids)+')')
        params.extend(sorted(extra_ids))
    if con.extra_blocks is not None:
        extra_all = extra_keys(rec)
        extra_allowed = [key for key in extra_all if con.extra_blocks.get(key,0)<=max_block]
        if extra_allowed:
            queries.append('SELECT rid FROM extra.blocks WHERE key IN ('+','.join('?' for _ in extra_allowed)+')')
            params.extend(extra_allowed)
        broad_extra=sorted((k for k in extra_all if max_block<con.extra_blocks.get(k,0)<=50000),
                           key=lambda k:(con.extra_blocks[k],k))
        words=[k for k in broad_extra if '|w|' in k]
        full_names=[k for k in broad_extra if '|f|' in k]
        pairs=[]
        if len(words)>1:pairs.append((words[0],words[1]))
        if len(full_names)>1:pairs.append((full_names[0],full_names[1]))
        if words and full_names:pairs.append((words[0],full_names[0]))
        new_ids=set()
        for first,second in pairs:
            if con.extra_blocks[first]>con.extra_blocks[second]:first,second=second,first
            found=con.execute('SELECT a.rid FROM extra.blocks a JOIN extra.blocks b ON a.rid=b.rid '
                              'WHERE a.key=? AND b.key=? LIMIT ?',(first,second,max_block+1)).fetchall()
            if len(found)<=max_block:new_ids.update(r[0] for r in found)
        if new_ids:
            queries.append('SELECT rid FROM records WHERE rid IN ('+','.join('?' for _ in new_ids)+')')
            params.extend(sorted(new_ids))
    pool = con.execute('SELECT eid,name,core,address,country FROM records WHERE rid IN ('+
                       ' UNION '.join(queries)+')',params).fetchall() if queries else []
    if not pool:
        return []
    names = process.cdist([rec[2]], [r[2] for r in pool], scorer=fuzz.token_sort_ratio, dtype=np.float64)[0]
    roman_names = process.cdist([roman_core(rec[2])], [roman_core(r[2]) for r in pool], scorer=fuzz.token_sort_ratio, dtype=np.float64)[0]
    names = np.maximum(names, roman_names)
    addresses = process.cdist([rec[3]], [r[3] for r in pool], scorer=fuzz.token_sort_ratio, dtype=np.float64)[0]
    address_sets = process.cdist([rec[3]], [r[3] for r in pool], scorer=fuzz.token_set_ratio, dtype=np.float64)[0]
    addresses = 0.55*addresses + 0.45*address_sets
    addresses *= np.asarray([bool(rec[3] and r[3]) for r in pool])
    numbers = np.asarray([100*overlap(numeric_tokens(rec[3]), numeric_tokens(r[3])) for r in pool])
    scores = np.maximum(0.6*names+0.3*addresses+0.1*numbers, 0.35*names+0.5*addresses+0.15*numbers)
    name_scores = 0.85*names+0.1*addresses+0.05*numbers
    ordered = sorted(range(len(pool)), key=lambda i: (-name_scores[i], pool[i][0]))[:min(10,top_k)]
    included = set(ordered)
    # Reserve address-focused slots for aliases and corrupted business names.
    address_scores = 0.8*addresses+0.15*numbers+0.05*names
    for i in sorted(range(len(pool)),key=lambda i:(-address_scores[i],pool[i][0]))[:min(15,top_k-len(ordered))]:
        if i not in included:
            ordered.append(i)
            included.add(i)
    for i in sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i][0])):
        if len(ordered) >= top_k:
            break
        if i not in included:
            ordered.append(i)
            included.add(i)
    return [pool[i] for i in ordered]


def prepare(args):
    """Sample anchors by hash; retain all their labels plus sampled distractors."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    profile = {}
    selected = {}
    required_targets = set()
    truth = {}
    for row in rows(data/'train/train_ground_truth.tsv'):
        eid = row['source1_entity_id']
        if bucket(eid) < args.anchor_rate * 10000:
            truth[eid] = row['matched_entity_ids']
            required_targets.update(filter(None, row['matched_entity_ids'].split(',')))
    log(f'Selected {len(truth):,} anchors and {len(required_targets):,} labeled targets')
    for split in ('train', 'test'):
        for source in (1, 2, 3):
            path = data/split/f'{split}_source{source}.tsv'
            countries, missing = Counter(), Counter()
            n = kept = 0
            output = open(out/f'train_source{source}.tsv', 'w', encoding='utf-8', newline='') if split == 'train' else None
            writer = csv.DictWriter(output, fieldnames=FIELDS, delimiter='\t', lineterminator='\n') if output else None
            if writer:
                writer.writeheader()
            for row in rows(path):
                n += 1
                countries[row['country']] += 1
                for field in FIELDS[1:]:
                    if not row[field].strip():
                        missing[field] += 1
                if split == 'train':
                    eid = row['entity_id']
                    take = eid in truth if source == 1 else (eid in required_targets or bucket(eid) < args.distractor_rate*10000)
                    if take:
                        writer.writerow(row)
                        kept += 1
                        if source == 1:
                            selected[eid] = row['country']
            if output:
                output.close()
            profile[str(path.relative_to(data))] = dict(rows=n, countries=dict(countries), missing=dict(missing), pilot_rows=kept)
            log(f'{split} source{source}: {n:,} rows, {dict(countries)}, pilot retained {kept:,}')
    assert set(truth) == set(selected), 'Ground truth anchors missing from source1'
    write_rows(out/'train_ground_truth.tsv',
               ({'source1_entity_id': k, 'matched_entity_ids': v} for k, v in truth.items()),
               ['source1_entity_id', 'matched_entity_ids'])
    profile['pilot'] = dict(anchors=len(truth), positive_targets=len(required_targets),
                           anchor_rate=args.anchor_rate, distractor_rate=args.distractor_rate,
                           singletons=sum(not v for v in truth.values()), countries=dict(Counter(selected.values())))
    Path(args.report).write_text(json.dumps(profile, indent=2))


def macro_score(truth, predictions):
    if not truth:
        raise ValueError('No evaluation anchors')
    scores = []
    for eid, actual in truth.items():
        predicted = predictions.get(eid, set())
        if not actual:
            scores.append(float(not predicted))
        else:
            scores.append(1.25*len(actual & predicted)/(0.25*len(actual)+len(predicted)))
    return float(np.mean(scores))


def evaluate(truth, pair_ids, probabilities, threshold):
    predictions = {}
    for (a, b), p in zip(pair_ids, probabilities):
        if p >= threshold:
            predictions.setdefault(a, set()).add(b)
    return macro_score(truth, predictions), predictions


def train(args):
    from xgboost import XGBClassifier
    started = time.monotonic()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = {r['source1_entity_id']: set(filter(None, r['matched_entity_ids'].split(',')))
             for r in rows(data/'train_ground_truth.tsv')}
    owners = {}
    for eid, targets in truth.items():
        for target in targets:
            if target in owners and owners[target] != eid:
                raise ValueError('Shared labeled target: split connected anchor groups before training')
            owners[target] = eid
    # Split entire anchor groups before making pairs; pair labels never select candidates.
    split = {eid: ('train' if bucket('split:'+eid, 10) < 6 else
                   'tune' if bucket('split:'+eid, 10) < 8 else 'holdout') for eid in truth}
    con = connect(args.db)
    if not con.execute('SELECT complete FROM metadata').fetchone()[0]:
        raise ValueError('Incomplete index')
    pool_size = con.execute('SELECT records FROM metadata').fetchone()[0]
    assert con.execute('SELECT retrieval_version FROM metadata').fetchone()[0] == INDEX_VERSION
    xs, ys, pair_ids = [], [], []
    counts, recalled = {}, {}
    anchor_countries = {}
    for i, row in enumerate(rows(data/'train_source1.tsv'), 1):
        rec = record(row)
        eid = rec[0]
        anchor_countries[eid] = row['country']
        found = candidates(con, rec, args.top_k, args.max_block)
        counts[eid] = len(found)
        recalled[eid] = len(truth[eid] & {r[0] for r in found})
        for other in found:
            xs.append(features(rec, other, con))
            ys.append(int(other[0] in truth[eid]))
            pair_ids.append((eid, other[0]))
        if i % 1000 == 0:
            log(f'Built pairs for {i:,} anchors; {len(xs):,} candidates')
    con.close()
    x, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8)
    del xs, ys
    masks = {s: np.array([split[eid] == s for eid, _ in pair_ids]) for s in ('train', 'tune', 'holdout')}
    if args.cache_pairs:
        np.save(out/'pair_features.npy', x)
        np.save(out/'pair_labels.npy', y)
        np.save(out/'pair_ids.npy', np.asarray(pair_ids, dtype=str))
    if len(set(y[masks['train']])) != 2:
        raise ValueError('Training requires positive and negative candidate pairs')
    model = XGBClassifier(n_estimators=args.trees, max_depth=args.depth, learning_rate=0.07,
                          subsample=0.85, colsample_bytree=0.9, tree_method='hist',
                          n_jobs=4, random_state=42, eval_metric='logloss')
    model.fit(x[masks['train']], y[masks['train']])
    probability = model.predict_proba(x)[:, 1]
    subset_truth = {s: {eid: t for eid, t in truth.items() if split[eid] == s} for s in masks}
    subset_pairs = {s: [p for p, keep in zip(pair_ids, mask) if keep] for s, mask in masks.items()}
    thresholds = np.arange(0.05, 0.981, 0.01)
    tuning = [(float(t), evaluate(subset_truth['tune'], subset_pairs['tune'],
                                probability[masks['tune']], t)[0]) for t in thresholds]
    threshold, tune_score = max(tuning, key=lambda item: (item[1], item[0]))
    model_kind = 'xgboost'
    comparisons = {'xgboost': {'tune_macro_f05': tune_score, 'threshold': threshold}}
    if args.compare_svm:
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        log('Benchmarking a standardized linear SVM on the same training pairs')
        scaler = StandardScaler().fit(x[masks['train']])
        svm = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=42)
        svm.fit(scaler.transform(x[masks['train']]), y[masks['train']])
        coefficient = svm.coef_[0] / scaler.scale_
        intercept = float(svm.intercept_[0] - np.dot(coefficient, scaler.mean_))
        margins = x @ coefficient + intercept
        svm_thresholds = np.unique(np.concatenate((np.linspace(-3, 3, 121),
                                   np.quantile(margins[masks['tune']], np.linspace(0, 1, 101)))))
        svm_tuning = [(float(t), evaluate(subset_truth['tune'], subset_pairs['tune'],
                                         margins[masks['tune']], t)[0]) for t in svm_thresholds]
        svm_threshold, svm_score = max(svm_tuning, key=lambda item: (item[1], item[0]))
        comparisons['linear_svm'] = dict(tune_macro_f05=svm_score, threshold=svm_threshold)
        (out/'linear_svm.json').write_text(json.dumps(dict(coefficient=coefficient.tolist(), intercept=intercept), indent=2))
        if svm_score > tune_score:
            model_kind, threshold, tune_score, probability = 'linear_svm', svm_threshold, svm_score, margins
        log(f'Tuning comparison: {comparisons}; selected {model_kind}')
    holdout_score, predicted = evaluate(subset_truth['holdout'], subset_pairs['holdout'],
                                       probability[masks['holdout']], threshold)
    reports = {}
    for s, labels in subset_truth.items():
        total_true = sum(map(len, labels.values()))
        reports[s] = dict(anchors=len(labels), pairs=int(masks[s].sum()),
                          true_links=total_true, candidate_recall=sum(recalled[eid] for eid in labels)/max(1,total_true),
                          mean_candidates=float(np.mean([counts[eid] for eid in labels])),
                          p95_candidates=float(np.percentile([counts[eid] for eid in labels], 95)),
                          reduction_ratio=1-sum(counts[eid] for eid in labels)/max(1,len(labels)*pool_size),
                          zero_candidate_anchors=sum(counts[eid] == 0 for eid in labels))
    holdout_truth = subset_truth['holdout']
    country_scores = {country: macro_score({eid: t for eid, t in holdout_truth.items()
                                            if anchor_countries[eid] == country}, predicted)
                      for country in sorted({anchor_countries[eid] for eid in holdout_truth})}
    errors = [{'source1_entity_id': eid, 'country': anchor_countries[eid],
               'false_positive_ids': sorted(predicted.get(eid, set())-actual),
               'false_negative_ids': sorted(actual-predicted.get(eid, set()))}
              for eid, actual in holdout_truth.items() if predicted.get(eid, set()) != actual]
    model.save_model(out/'model.ubj')
    metadata = dict(threshold=threshold, top_k=args.top_k, max_block=args.max_block,
                    model_kind=model_kind,
                    trees=args.trees, depth=args.depth,
                    retrieval_version=RETRIEVAL_VERSION,
                    feature_names=FEATURE_NAMES, model_license='MIT', library_license='Apache-2.0',
                    indexed_target_records=pool_size,
                    training_scope=str(data.resolve()), refit_on_holdout=False)
    (out/'model_config.json').write_text(json.dumps(metadata, indent=2))
    report = dict(scope='Sampled labeled anchors; see indexed_target_records for retrieval pool size. No France validation labels. Not a leaderboard estimate.',
                  indexed_target_records=pool_size,
                  splits=reports, threshold=threshold, tune_macro_f05=tune_score,
                  selected_model=model_kind, model_comparison=comparisons,
                  holdout_macro_f05=holdout_score, holdout_by_country=country_scores,
                  holdout_empty_prediction_baseline=macro_score(holdout_truth, {}),
                  holdout_singletons=sum(not t for t in holdout_truth.values()),
                  seconds=time.monotonic()-started,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform == 'darwin' else 1024),
                  feature_importance=dict(zip(FEATURE_NAMES, map(float, model.feature_importances_))))
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    (out/'holdout_errors.json').write_text(json.dumps(errors[:200], indent=2))
    log(json.dumps(report, indent=2))


def predict(args):
    """Stream anchors and write the exact candidate set scored by the model."""
    from xgboost import XGBClassifier
    config = json.loads((Path(args.model)/'model_config.json').read_text())
    assert config['feature_names'] == FEATURE_NAMES, 'Feature definition changed'
    assert config['retrieval_version'] == RETRIEVAL_VERSION, 'Retrieval version changed'
    if config.get('model_kind') == 'linear_svm':
        model = json.loads((Path(args.model)/'linear_svm.json').read_text())
    else:
        model = XGBClassifier()
        model.load_model(Path(args.model)/'model.ubj')
        model.set_params(n_jobs=4)
    con = connect(args.db)
    assert con.execute('SELECT complete,retrieval_version FROM metadata').fetchone() == (1, INDEX_VERSION)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Partial files cannot be mistaken for a completed submission.
    paths = [out/'matching_results.tsv.partial', out/'candidate_pairs.tsv.partial']
    with open(paths[0], 'w') as mf, open(paths[1], 'w') as cf:
        mf.write('source1_entity_id\tmatched_entity_ids\n')
        cf.write('source1_entity_id\tcandidate_entity_ids\n')
        for i, row in enumerate(rows(args.source1), 1):
            rec = record(row)
            found = candidates(con, rec, config['top_k'], config['max_block'])
            x = np.asarray([features(rec, b, con) for b in found], dtype=np.float32)
            if not found:
                probabilities = []
            elif config.get('model_kind') == 'linear_svm':
                probabilities = x @ np.asarray(model['coefficient']) + model['intercept']
            else:
                probabilities = model.predict_proba(x)[:, 1]
            matched = [b[0] for b, p in zip(found, probabilities) if p >= config['threshold']]
            cf.write(rec[0]+'\t'+','.join(b[0] for b in found)+'\n')
            mf.write(rec[0]+'\t'+','.join(matched)+'\n')
            if i % 10000 == 0:
                log(f'Predicted {i:,} anchors')
    con.close()
    for path in paths:
        path.rename(path.with_suffix(''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare-pilot')
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='artifacts/pilot_data')
    p.add_argument('--report', default='reports/data_profile.json')
    p.add_argument('--anchor-rate', type=float, default=0.0025)
    p.add_argument('--distractor-rate', type=float, default=0.02)
    p.set_defaults(func=prepare)
    p = subs.add_parser('index')
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--db', required=True)
    p.set_defaults(func=lambda a: build_index(a.sources, a.db))
    p = subs.add_parser('train')
    p.add_argument('--data', required=True, help='Directory containing sampled train TSVs')
    p.add_argument('--db', required=True)
    p.add_argument('--out', default='artifacts/pilot_model')
    p.add_argument('--top-k', type=int, default=60)
    p.add_argument('--max-block', type=int, default=1000)
    p.add_argument('--compare-svm', action='store_true')
    p.add_argument('--cache-pairs', action='store_true')
    p.add_argument('--trees', type=int, default=220)
    p.add_argument('--depth', type=int, default=5)
    p.set_defaults(func=train)
    p = subs.add_parser('predict')
    p.add_argument('--source1', required=True)
    p.add_argument('--db', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='output')
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
