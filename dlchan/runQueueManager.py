from filelock import FileLock
from modules.runqueue import RunQueue

addToQueue = False
clearQueue = False

runQueueFN = 'data/runqueue.pickle'
runQueueLK = 'data/runqueue.lock'


mdlList = ['modelTmp.h5', 'cnn2d4-3_2000.h5', 'cnn4-5000.h5', 'cnn2.h5', 'cnn2d4-3_7000.h5', 'cnn1.h5']

q = RunQueue(runQueueFN, FileLock(runQueueLK, thread_local=False))

if clearQueue:
    q.clear()

if addToQueue:
    for m in mdlList:
        q.add(m)

mdlListRead = q.waiting(date=True)
for m in mdlListRead:
    print(m[1].strftime("%Y%m%d-%H:%M:%S"), m[0])
