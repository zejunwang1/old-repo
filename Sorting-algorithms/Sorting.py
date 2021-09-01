
# pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ sklearn

import sys
import time
import random
sys.setrecursionlimit(2000)

class Sort:
    def bubbleSort(self, seq):
        n = len(seq)
        for i in range(n - 1):
            for j in range(n - i - 1):
                if seq[j] > seq[j + 1]:
                    seq[j], seq[j + 1] = seq[j + 1], seq[j]
        return

    def insertionSort(self, seq, n):
        if n == 1:
            return
        insertionSort(seq, n - 1)
        i = n - 2
        while (seq[i] > seq[i + 1]) & (i >= 0):
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
            i -= 1
        return

    def insertionSort(self, seq):
        n = len(seq)
        if n <= 1: 
            return
        for i in range(n - 1):
            j = i + 1
            start = i;
            if seq[start] < seq[j]:
                continue
            while (start >= 0) & (seq[start] > seq[start + 1]):
                seq[start], seq[start + 1] = seq[start + 1], seq[start]
                start -= 1
        return

    def selectionSort(self, seq):
        n = len(seq)
        for i in range(n - 1):
            max_idx = n - i - 1
            for j in range(n - i - 1):
                if seq[j] > seq[max_idx]:
                    max_idx = j
            if max_idx != n - i - 1:
                seq[max_idx], seq[n - i - 1] = seq[n - i - 1], seq[max_idx]
        return

    def mergeSort(self, seq, lo, hi):
        if hi - lo == 1:
            return
        mid = int((lo + hi) / 2)
        self.mergeSort(seq, lo, mid)
        self.mergeSort(seq, mid, hi)
        a = list()
        b = list()
        for i in range(mid - lo):
            a.append(seq[lo + i])
        for i in range(hi - mid):
            b.append(seq[mid + i])
        i = 0
        j = 0
        k = lo
        while (i < mid - lo) & (j < hi - mid):
            if a[i] < b[j]:
                seq[k] = a[i]
                k += 1
                i += 1
            else:
                seq[k] = b[j]
                k += 1
                j += 1
        if i == mid - lo:
            while j < hi - mid:
                seq[k] = b[j]
                k += 1
                j += 1
        elif j == hi - mid:
            while i < mid - lo:
                seq[k] = a[i]
                k += 1
                i += 1
        return

    def quickSort(self, seq, lo, hi):
        if (hi - lo == 1) | (lo >= hi):
            return
        partition_pos = self.partition(seq, lo, hi)
        self.quickSort(seq, lo, partition_pos)
        self.quickSort(seq, partition_pos + 1, hi)
        return

    def partition(self, seq, lo, hi):
        i = lo + 1
        j = hi - 1
        while i <= j:
            if seq[i] > seq[lo]:
                seq[i], seq[j] = seq[j], seq[i]
                j -= 1
            else:
                i += 1
        seq[lo], seq[j] = seq[j], seq[lo]
        return j

    def heapSort(self, seq):
        n = len(seq)
        i = int(n / 2)
        for k in range(i):
            j = i - k - 1
            self.adjustHeap(seq, j, n)
        for k in range(0, n):
            seq[0], seq[n - k - 1] = seq[n - k - 1], seq[0]
            self.adjustHeap(seq, 0, n - k - 1)
        return

    def adjustHeap(self, seq, i, n):
        while (1):
            p = i
            l = 2 * i + 1
            r = 2 * i + 2
            #if (l < n) & (seq[l] > seq[p]):
            #    p = l
            #if (r < n) & (seq[r] > seq[p]):
            #    p = r
            if l < n:
                if seq[l] > seq[p]:
                    p = l
            if r < n:
                if seq[r] > seq[p]:
                    p = r
            if p != i:
                seq[p], seq[i] = seq[i], seq[p]
                i = p
            else:
                break
        return

    def shellSort(self, seq):
        n = len(seq)
        if n <= 1:
            return
        gap = int(n / 2)
        while gap > 0:
            for i in range(gap, n):
                j = i - gap
                while (j >= 0) & (seq[j] > seq[j + gap]):
                    seq[j], seq[j + gap] = seq[j + gap], seq[j]
                    j -= gap
            gap = int(gap / 2)
        return

    def countingSort(self, seq):
        if len(seq) <= 1:
            return
        seq_min = seq[0]
        seq_max = seq[0]
        for i in range(1, len(seq)):
            if seq[i] < seq_min:
                seq_min = seq[i]
            if seq[i] > seq_max:
                seq_max = seq[i]
        counting = [0] * (seq_max - seq_min + 1)
        for i in range(len(seq)):
            counting[seq[i] - seq_min] += 1
        i = 0
        index = 0
        while i < (seq_max - seq_min + 1):
            if counting[i] != 0:
                seq[index] = i + seq_min
                index += 1
                counting[i] -= 1
            else:
                i += 1
        return

    def bucketSort(self, seq, bucket_count):
        if len(seq) <= 1:
            return
        seq_min = seq[0]
        seq_max = seq[0]
        for i in range(1, len(seq)):
            if seq[i] < seq_min:
                seq_min = seq[i]
            if seq[i] > seq_max:
                seq_max = seq[i]
        buckets = [ [] for i in range(bucket_count) ]
        for i in range(len(seq)):
            buckets_index = int(float(seq[i] - seq_min) / float(seq_max - seq_min + 1) * bucket_count)
            self.appendElement(buckets[buckets_index], seq[i])
        k = 0
        for i in range(len(buckets)):
            for j in range(len(buckets[i])):
                seq[k] = buckets[i][j]
                k += 1
        return

    def appendElement(self, bucket, val):
        if not bucket:
            bucket.append(val)
            return
        bucket.append(val)
        for i in range(len(bucket) - 1):
            j = len(bucket) - i - 2
            if bucket[j] > bucket[j + 1]:
                bucket[j], bucket[j + 1] = bucket[j + 1], bucket[j]
        return

    def radixSort(self, seq):
        if len(seq) <= 1:
            return
        seq_max = seq[0]
        for i in range(1, len(seq)):
            if (seq[i] > seq_max):
                seq_max = seq[i]
        bit_num = 0
        while seq_max > 0:
            seq_max = int(seq_max / 10)
            bit_num += 1
        bit_array = [ [] for i in range(10) ]
        bit = 0
        n = 1
        k = 0
        while bit < bit_num:
            for i in range(len(seq)):
                bit_idx = int(seq[i] / n) % 10
                bit_array[bit_idx].append(seq[i])
            for i in range(10):
                if bit_array[i]:
                    for j in range(len(bit_array[i])):
                        seq[k] = bit_array[i][j]
                        k += 1
                bit_array[i].clear()
            bit += 1
            n *= 10
            k = 0
        return


if __name__ == '__main__':
    n = 50000
    num = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for i in range(len(num)) :
	    seq = []
	    for j in range(num[i]) :
		    seq.append(random.randint(1, n))

	    print("sorting sequence number:  %d ......" % num[i])

	    s = Sort()

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.bubbleSort(sequence)
	    time2 = time.clock()
	    sequence.clear()
	    print("bubble sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.insertionSort(sequence)
	    time2 = time.clock()
	    sequence.clear()
	    print("insertion sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.selectionSort(sequence)
	    time2 = time.clock()
	    sequence.clear()
	    print("selection sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.mergeSort(sequence, 0, len(sequence))
	    time2 = time.clock()
	    sequence.clear()
	    print("merge sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.quickSort(sequence, 0, len(sequence))
	    time2 = time.clock()
	    sequence.clear()
	    print("quick sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.heapSort(sequence)
	    time2 = time.clock()
	    sequence.clear()
	    print("heap sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.shellSort(seq)
	    time2 = time.clock()
	    sequence.clear()
	    print("shell sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.countingSort(seq)
	    time2 = time.clock()
	    sequence.clear()
	    print("counting sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.bucketSort(seq, 50)
	    time2 = time.clock()
	    sequence.clear()
	    print("bucket sort time cost:  %.3f" % (time2 - time1))

	    sequence = [seq[i] for i in range(len(seq))]
	    time1 = time.clock()
	    s.radixSort(seq)
	    time2 = time.clock()
	    sequence.clear()
	    print("radix sort time cost:  %.3f" % (time2 - time1))

	    print("\n\n\n")






