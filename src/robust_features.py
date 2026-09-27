"""Corpus-weighted text similarities and canonical address comparisons."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
import joblib
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from pipeline import roman_core, numeric_tokens, log

# Text aliases only; these are not business identity or geographic lookup services.
US_STATES='alabama:al|alaska:ak|arizona:az|arkansas:ar|california:ca|colorado:co|connecticut:ct|delaware:de|district of columbia:dc|florida:fl|georgia:ga|hawaii:hi|idaho:id|illinois:il|indiana:in|iowa:ia|kansas:ks|kentucky:ky|louisiana:la|maine:me|maryland:md|massachusetts:ma|michigan:mi|minnesota:mn|mississippi:ms|missouri:mo|montana:mt|nebraska:ne|nevada:nv|new hampshire:nh|new jersey:nj|new mexico:nm|new york:ny|north carolina:nc|north dakota:nd|ohio:oh|oklahoma:ok|oregon:or|pennsylvania:pa|rhode island:ri|south carolina:sc|south dakota:sd|tennessee:tn|texas:tx|utah:ut|vermont:vt|virginia:va|washington:wa|west virginia:wv|wisconsin:wi|wyoming:wy'
INDIA_STATES='andhra pradesh:ap|arunachal pradesh:ar|assam:as|bihar:br|chhattisgarh:ct|goa:ga|gujarat:gj|haryana:hr|himachal pradesh:hp|jharkhand:jh|karnataka:ka|kerala:kl|madhya pradesh:mp|maharashtra:mh|maharastra:mh|manipur:mn|meghalaya:ml|mizoram:mz|nagaland:nl|odisha:od|orissa:od|punjab:pb|rajasthan:rj|sikkim:sk|tamil nadu:tn|tamilnadu:tn|telangana:tg|tripura:tr|uttar pradesh:up|uttarakhand:ut|uttaranchal:ut|west bengal:wb|delhi:dl'
ALIASES={country:dict(item.split(':') for item in text.split('|')) for country,text in [('us',US_STATES),('india',INDIA_STATES)]}
PATTERNS={country:re.compile(r'\b(?:'+'|'.join(re.escape(k) for k in sorted(values,key=len,reverse=True))+r')\b') for country,values in ALIASES.items()}
LEGAL={'sa','sarl','sas','sasu','eurl','sci','com','net','org'}
ROBUST_NAMES=['canonical_name_ratio','canonical_name_sorted','canonical_name_set','canonical_name_partial',
              'canonical_address_ratio','canonical_address_sorted','canonical_address_set','canonical_address_partial',
              'canonical_address_letters_sorted','canonical_address_letters_set',
              'name_character_tfidf_cosine','address_character_tfidf_cosine',
              'canonical_name_token_containment','canonical_address_token_containment',
              'canonical_address_first_number_similarity','canonical_address_number_count_ratio',
              'address_tfidf_seed_similarity','address_tfidf_seed_name_similarity','address_tfidf_seed_probability',
              'name_tfidf_seed_similarity','name_tfidf_seed_address_similarity','name_tfidf_seed_probability']


@lru_cache(maxsize=50000)
def canonical_name(core,country):
    value=roman_core(core)
    return ' '.join(t for t in value.split() if t not in LEGAL)


@lru_cache(maxsize=50000)
def canonical_address(address,country):
    # roman_core is unsuitable for addresses because it removes company tokens.
    from anyascii import anyascii
    text=anyascii(address).lower()
    text=re.sub(r'\b([0-9]+)(?:st|nd|rd|th)\b',r'\1',text)
    text=re.sub(r'\b[0-9]+\b',lambda m:str(int(m.group())),text)
    if country in PATTERNS:text=PATTERNS[country].sub(lambda m:ALIASES[country][m.group()],text)
    return ' '.join(re.findall('[a-z0-9]+',text))


class RobustFeatures:
    def __init__(self,path):
        self.vectorizers=joblib.load(path)

    def batch(self,anchors,found_groups,probability_groups):
        records=[];positions={}
        for a,found in zip(anchors,found_groups):
            for r in [a]+found:
                if r[0] not in positions:positions[r[0]]=len(records);records.append(r)
        names=[canonical_name(r[2],r[4]) for r in records]
        addresses=[canonical_address(r[3],r[4]) for r in records]
        nv=self.vectorizers['name'].transform(names)
        av=self.vectorizers['address'].transform(addresses)
        result=[]
        for a,found,prob in zip(anchors,found_groups,probability_groups):
            count=len(found);values=np.zeros((count,len(ROBUST_NAMES)),dtype=np.float32)
            if not count:result.append(values);continue
            ai=positions[a[0]];bi=np.array([positions[r[0]] for r in found]);an=names[ai];aa=addresses[ai]
            ns=nv[bi];ads=av[bi]
            values[:,10]=(ns@nv[ai].T).toarray().ravel()
            values[:,11]=(ads@av[ai].T).toarray().ravel()
            for i,j in enumerate(bi):
                bn=names[j];ba=addresses[j]
                values[i,:4]=np.asarray([fuzz.ratio(an,bn),fuzz.token_sort_ratio(an,bn),fuzz.token_set_ratio(an,bn),fuzz.partial_ratio(an,bn)])/100 if an and bn else 0
                if aa and ba:
                    values[i,4:8]=np.asarray([fuzz.ratio(aa,ba),fuzz.token_sort_ratio(aa,ba),fuzz.token_set_ratio(aa,ba),fuzz.partial_ratio(aa,ba)])/100
                    al=re.sub('[0-9]+','',aa);bl=re.sub('[0-9]+','',ba)
                    values[i,8:10]=np.asarray([fuzz.token_sort_ratio(al,bl),fuzz.token_set_ratio(al,bl)])/100
                left,right=set(an.split()),set(bn.split());values[i,12]=len(left&right)/max(1,min(len(left),len(right)))
                left,right=set(aa.split()),set(ba.split());values[i,13]=len(left&right)/max(1,min(len(left),len(right)))
                left,right=re.findall('[0-9]+',aa),re.findall('[0-9]+',ba)
                values[i,14]=fuzz.ratio(left[0],right[0])/100 if left and right else 0
                values[i,15]=min(len(left),len(right))/max(1,len(left),len(right))
            if count>1:
                nm=(ns@ns.T).toarray();am=(ads@ads.T).toarray();np.fill_diagonal(nm,0);np.fill_diagonal(am,0)
                for offset,similarity,other in ((16,am,nm),(19,nm,am)):
                    seed=np.argmax(similarity*np.asarray(prob)[None,:],axis=1);indices=np.arange(count)
                    values[:,offset]=similarity[indices,seed]
                    values[:,offset+1]=other[indices,seed]
                    values[:,offset+2]=np.asarray(prob)[seed]
            result.append(values)
        return result


def fit(databases,destination):
    documents={'name':[],'address':[]};counts={}
    for database in databases:
        con=sqlite3.connect(database)
        for core,address,country in con.execute('SELECT core,address,country FROM records WHERE rid % 97 = 0'):
            documents['name'].append(canonical_name(core,country));documents['address'].append(canonical_address(address,country))
            counts[country]=counts.get(country,0)+1
        con.close()
    models={}
    for field in documents:
        log(f'Fit {field} character weights on {len(documents[field]):,} supplied unlabeled records')
        model=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=300000,sublinear_tf=True,dtype=np.float32)
        model.fit(documents[field]);models[field]=model
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    joblib.dump(models,destination,compress=3)
    destination.with_suffix('.json').write_text(json.dumps(dict(records_by_country=counts,scope='Unlabeled systematic sample from supplied training/test target records. Character frequencies only; no entity labels.'),indent=2))
    log('Saved corpus character weights: '+str(destination))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--db',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();fit(a.db,a.out)
