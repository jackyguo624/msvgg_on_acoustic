from __future__ import print_function
from multiprocessing import Process, Value, Queue, Lock
import time
import sys
import kaldi_io
import torch

def warning(*obj):
    print("WARNING: ", *obj, file=sys.stderr)


class Dataloader(object):

    def __init__(self,feature,ali, ext_data_n=20,batch_size=256):
        self.no_ext_data = Value('b', 0)
        self.qsize = Value('i', 0)
        self.data_queue = Queue()
        self.ext_data_n = ext_data_n
        self.ali = ali
        self.feature = feature
        self.ali_dict = self.read_ali(ali)
        self.batch_size = batch_size
        self.lock = Lock()
        p = Process(target=self.reader, args=(self.no_ext_data,self.qsize, self.data_queue, self.lock))
        p.daemon=True
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
#                self.lock.acquire()
                tt = self.data_queue.get()
                assert(tt is not None)
                a.append(tt)
                self.qsize.value -= 1
#                self.lock.release()
                i = i + 1
            elif self.no_ext_data.value == 1:
                i_list, l_list = zip(*a)
                return torch.FloatTensor(i_list),torch.LongTensor(l_list)
            else:
                warning('Fetch nothing queue size {} sleep'.format(self.qsize.value))
                time.sleep(5)
                continue
        warning('Get batch queue size {}'.format(self.qsize.value))
        i_list, l_list = zip(*a)
        return torch.FloatTensor(i_list), torch.LongTensor(l_list)

    def __len__(self):
        return self.qsize.value

    @staticmethod
    def read_ali(ali):
        filename = ali #'/home/jiaqi/OKR/ensembleVGG/ali/ali.ark'
        return {k: m for k, m in kaldi_io.read_ali_ark(filename)}

    def reader(self, no_ext_data, qsize, data_queue, lock):
        i=1
        filename = self.feature.format(i) #'/home/jiaqi/OKR/ensembleVGG/mfcc40_23/feats/expand_feats.{0:d}.ark'.format(i)

        while i <= self.ext_data_n:
            if 100 < qsize.value:
                warning("Queue Size : {} sleep.".format(qsize.value))
                time.sleep(5)
                continue

            warning("Read {} {} feats".format(filename,i))
            orig = {k: m for k, m in kaldi_io.read_mat_ark(filename)}
#            lock.acquire()
            count = 0
            for key in orig.keys():
                feature_mat = orig[key].tolist()
                if key not in self.ali_dict.keys():
                    continue
                label_list = self.ali_dict[key].tolist()
                assert (len(feature_mat) == len(label_list))
                for f,l in zip(feature_mat,label_list):
                    data_queue.put((f, l))
                count += len(label_list)
            i = i + 1
            qsize.value += count
            warning('after adding. queue size {}'.format(qsize.value))
#            lock.release()
            filename = self.feature.format(i) # '/home/jiaqi/OKR/ensembleVGG/mfcc40_23/feats/expand_feats.{0:d}.ark'.format(i)
        no_ext_data.value = 1
