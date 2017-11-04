from __future__ import print_function
from multiprocessing import Process, Value,Queue, Lock
import time
import sys
import kaldi_io


def warning(*obj):
    print("WARNING: ", *obj, file=sys.stderr)


class Dataloader(object):

    def __init__(self, ext_data_n=20):
        self.no_ext_data = Value('b', 0)
        self.qsize = Value('i', 0)
        self.data_queue = Queue()
        self.ext_data_n = ext_data_n
        self.ali_dict = self.read_ali()
        self.batch_size = 50
        self.lock = Lock()
        p = Process(target=self.reader, args=(self.no_ext_data,self.qsize, self.data_queue, self.lock))
        p.start()

    def __iter__(self):
        return self

    def next(self):
        if self.no_ext_data.value == 1 and self.qsize.value == 0:
            raise StopIteration

        i = 1
        a = list()
        while i <= self.batch_size:
            if self.qsize.value > 0:
                self.lock.acquire()
                tt = self.data_queue.get()
                assert(tt is not None)
                a.append(tt)
                self.qsize.value -= 1
                self.lock.release()
                i = i + 1
            elif self.no_ext_data.value == 1:
                return a
            else:
                warning('Fetch nothing queue size {} sleep'.format(self.qsize.value))
                time.sleep(5)
                continue
        #warning('Get batch queue size {}'.format(self.qsize.value))
        return a

    @staticmethod
    def read_ali():
        filename = '/home/jiaqi/OKR/ensembleVGG/ali/ali.ark'
        return {k: m for k, m in kaldi_io.read_ali_ark(filename)}

    def reader(self, no_ext_data, qsize, data_queue, lock):
        i=1
        filename = '/home/jiaqi/OKR/ensembleVGG/mfcc40_23/feats/expand_feats.{0:d}.ark'.format(i)

        while i <= self.ext_data_n:
            if 200 < qsize.value:
                warning("Queue Size : {} sleep.".format(qsize.value))
                time.sleep(5)
                continue

            warning("Read file {} feats".format(i))
            orig = {k: m for k, m in kaldi_io.read_mat_ark(filename)}
            lock.acquire()
            for key in orig.keys():
                data_queue.put((orig[key], self.ali_dict[key]))
            i = i + 1
            orig_len = len(orig)
            qsize.value += orig_len
            warning('after adding. queue size {}'.format(qsize.value))
            lock.release()
            filename = '/home/jiaqi/OKR/ensembleVGG/mfcc40_23/feats/expand_feats.{0:d}.ark'.format(i)
        no_ext_data.value = 1
