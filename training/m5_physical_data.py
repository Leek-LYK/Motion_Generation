"""Read-only subset view of the independently audited physical-v2 cache."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np,pandas as pd,torch
from training.m3 import INPUTS


class PhysicalSubsetDataset:
    def __init__(self,entries,input_dir,stats_dir=None,cache_size=0):
        self.entries=entries;self.directory=Path(input_dir);self.config=json.loads((self.directory/'config.json').read_text());self.ready=json.loads((self.directory/'READY').read_text())
        if not self.ready['passed'] or self.ready['fingerprint']!=self.config['fingerprint']:raise ValueError('Physical cache identity mismatch')
        rows=pd.read_parquet(self.directory/'samples.parquet').to_dict('records');self.by_id={r['sample_id']:r for r in rows}
        if not {r['sample_id'] for r in entries}<=set(self.by_id):raise ValueError('Subset missing from physical cache')
        stats=Path(stats_dir or self.directory/'stats/motion');mean=np.concatenate([np.load(stats/g/'mean.npy') for g in ('global_root','body')]);std=np.concatenate([np.load(stats/g/'std.npy') for g in ('global_root','body')])
        self.mean=mean.astype(np.float32);self.scale=np.sqrt(std.astype(np.float32)**2+np.float32(1e-5))
        self.features=np.load(self.directory/'features.npy',mmap_mode='r',allow_pickle=False)
        self.text=np.load(self.config['text_cache'],mmap_mode='r',allow_pickle=False);self.cache=OrderedDict();self.cache_size=cache_size

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        entry=self.entries[index];row=self.by_id[entry['sample_id']]
        for key in ('text_id','text','motion_id','split'):
            if entry[key]!=row[key]:raise ValueError(f'Subset identity mismatch: {key}')
        n=int(row['num_frames']);a=int(row['frame_offset']);raw=np.asarray(self.features[a:a+n]);x=(raw-self.mean)/self.scale
        text=np.array(self.text[int(row['text_row'])],copy=True)
        if x.shape!=(n,417) or text.shape!=(1,4096) or not np.isfinite(x).all():raise ValueError('Malformed physical input')
        sample={'x0':torch.from_numpy(x.copy()),'text_feat':torch.from_numpy(text),'first_heading_angle':torch.tensor(row['first_heading_angle'],dtype=torch.float32),
          'length':n,'sample_id':row['sample_id'],'text_id':row['text_id'],'motion_id':row['motion_id'],'text':row['text'],'split':row['split'],
          'origin_xz':torch.tensor([row['origin_x'],row['origin_z']],dtype=torch.float32),'group':entry['group'],'cohort':entry.get('cohort','train')}
        if self.cache_size:
            self.cache[index]=sample
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return sample

    def batch(self,indices):return [self[int(i)] for i in indices]
