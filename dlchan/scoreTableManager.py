from datetime import datetime
from filelock import FileLock
from modules.score import ScoreTable


firstN = 0
write  = False

tableFN = 'data/scoretable.pickle'
tableLK = 'data/scoretable.lock'

tbl = ScoreTable(tableFN, FileLock(tableLK, thread_local=False))


if firstN > 0:
    tbl._table = {k: tbl._table[k] for k in list(tbl._table)[:firstN]}
    if write:
        tbl._ScoreTable__write()


for s in tbl._table:
    print(tbl._table[s][3].strftime("%Y-%m-%d %H:%M:%S"), s, tbl._table[s][:3])
print('table updated on:', tbl.updateDate().strftime("%Y-%m-%d %H:%M:%S"))