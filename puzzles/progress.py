import time

class Progress:
    def __init__(self, total_iter=None, label='default', response_time = 60, on=True):
        self.label = label
        if total_iter is not None:
            self.tot = total_iter * 1.0
        else:
            self.tot = None
        self.rt = response_time
        self.cnt = 0
        self.start = None
        self.on = on
        self.time0 = None

    def count(self):
        if self.on is False:
            return 0
        self.cnt += 1
        if self.start is None:
            self.time0 = time.time()
            self.start = self.time0
        else:
            end = time.time()
            if end - self.start > self.rt:
                if self.tot is None:
                    print(self.label, 'progress:\t%d\ttime:\t%0.1fs' % (self.cnt, end - self.time0))
                else:
                    print(self.label, 'progress:\t%0.4f%%\ttime:\t%0.1fs' % (self.cnt/self.tot * 100, end - self.time0))
                self.start = end




