def read_h5(path):
    import h5py
    import tqdm
    import pandas as pd
    res={}
    a=h5py.File(path)
    for k,v in tqdm.tqdm(list(a.items()),desc='数据加载中……'):
        try:
            v=list(v.values())[-1]
        except Exception:
            res[k]=pd.DataFrame(v)
    return res