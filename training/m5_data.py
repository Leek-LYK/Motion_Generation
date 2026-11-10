"""Immutable M5 subset views over the frozen M2 mmap dataset."""
import hashlib
import re
from pathlib import Path
from collections import OrderedDict
from data_pipeline.model_input_dataset import G1ModelInputDataset

GROUPS=('walk','jog','jump','raise_arms','turn','squat')
PATTERNS={
 'walk':r'\bwalk(?:s|ing)?\b',
 'jog':r'\b(?:jog(?:s|ging)?|run(?:s|ning)?)\b',
 'jump':r'\bjump(?:s|ing)?\b',
 'raise_arms':r'\b(?:rais(?:e|es|ing)|lift(?:s|ing)?)\b(?:\s+[\w-]+){0,5}\s+(?:arms|hands)\b',
 'turn':r'\b(?:turn(?:s|ing)?|rotat(?:e|es|ing))\b',
 'squat':r'\b(?:squat(?:s|ting)?|crouch(?:es|ed|ing)?)\b',
}
EXCLUDE=r'\b(?:crate|cupboard|flute|recorder|nail|fence|valve|burger|danc\w*|twirl\w*|object|chainsaw|table|chair|wall|ladder|stair\w*|box|rope|gun|weapon|platform|obstacle|cigarette|smok\w*|climb\w*|ledge|bench|stool|fall(?:s|ing)?|somersault|cartwheel|roll(?:s|ing)?|handstand|inverted|baby|fishing|crank|lever|item|handle|door|book|sit(?:s|ting)?|seated|kneel(?:s|ing)?|crawl(?:s|ing)?|monkey|vogue|fire|hip[- ]hop|page|paper|ball|bottle|cup|stick|sword|bag|bucket|basket|phone|pole)\b'


def classify(text):
    text=text.lower()
    if re.search(EXCLUDE,text):return None,[]
    matches=[g for g,p in PATTERNS.items() if re.search(p,text)]
    primary=[g for g in matches if g in GROUPS[:4]]
    if len(primary)>1:return None,matches
    if primary:
        if primary[0]=='raise_arms' and 'lifting their hands off their head' in text:return None,matches
        return primary[0],matches
    if matches==['turn'] and re.search(r'\b(?:turn(?:s|ing)?|rotat(?:e|es|ing))\s+(?:(?:their|the|both|a|left|right|own)\s+){0,4}(?:head|hands?|arms?|wrists?|feet|foot|ankles?|palms?|fingers?|neck)\b',text):
        return None,matches
    if matches==['turn'] and re.search(r'\b(?:head|hands?|arms?|wrists?|palms?|fingers?|neck|feet)\s+(?:(?:is|are|slowly|quickly|gradually|steadily)\s+){0,2}(?:turn(?:s|ing)?|rotat(?:e|es|ing))\b',text):
        return None,matches
    if len(matches)==1:
        if matches[0]=='squat':matches=matches+['static_crouch' if re.search(r'\b(?:remains?|idle|is in|in a crouch position)\b',text) else 'crouch_transition_or_style']
        return matches[0],matches
    return None,matches


def rank(value,salt='m5-data-5101'):
    return hashlib.sha256((salt+':'+str(value)).encode()).hexdigest()


def group_keys(row):
    return {(key,str(row[key])) for key in ('source_group_id','validation_group_id','motion_id')
            if row.get(key) is not None and str(row[key]) not in ('','nan','None')}


class SubsetDataset:
    """Bounded LRU accelerates short diagnostic runs without copying all M2 arrays."""
    def __init__(self,entries,cache_size=0):
        self.entries=entries;self.datasets={};self.indices={};self.cache_size=cache_size;self.cache=OrderedDict()
        for split in sorted({r['split'] for r in entries}):
            ds=G1ModelInputDataset(split);self.datasets[split]=ds
            wanted={r['sample_id'] for r in entries if r['split']==split}
            self.indices[split]={r['sample_id']:i for i,r in enumerate(ds.rows) if r['sample_id'] in wanted}
            if set(self.indices[split])!=wanted:raise ValueError('Missing M2 events')

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        row=self.entries[index];sample=self.datasets[row['split']][self.indices[row['split']][row['sample_id']]]
        for key in ('text_id','text','motion_id'):
            if sample[key]!=row[key]:raise ValueError(f'M2 identity mismatch: {key}')
        sample=dict(sample,group=row['group'],cohort=row.get('cohort','train'))
        if self.cache_size:
            self.cache[index]=sample
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return sample

    def batch(self,indices):return [self[int(i)] for i in indices]
