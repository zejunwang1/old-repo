
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>

void seqGenerate(std::vector<int>& seq, int n);
void bubbleSort(std::vector<int>& seq);
void insertionSort(std::vector<int>& seq);
void insertionSort(std::vector<int>& seq, int n);
void selectionSort(std::vector<int>& seq);
void mergeSort(std::vector<int>& seq, int lo, int hi);
void quickSort(std::vector<int>& seq, int lo, int hi);
int partition(std::vector<int>& seq, int lo, int hi);
void heapSort(std::vector<int>& seq);
void adjustHeap(std::vector<int>& seq, int i, int n);
void shellSort(std::vector<int>& seq);
void countingSort(std::vector<int>& seq);
void bucketSort(std::vector<int>& seq, int bucket_count);
void appendElement(std::vector<int>& bucket, int val);
void radixSort(std::vector<int>& seq);

int main()
{
	int method = 10;
	std::vector<int> n_seq{ 10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };
	const int seq_num = n_seq.size();
	double** total_time;
	total_time = new double* [seq_num];
	clock_t start, end;
	for (int i = 0; i < seq_num; i++)
	{
		*(total_time + i) = new double[method];
	}
	for (int i = 0; i < seq_num; i++)
	{
		std::vector<int> seq;
		seqGenerate(seq, n_seq[i]);
		
		std::vector<int> sequence;

		std::cout << "sorting sequence number: " << n_seq[i] << " ......" << std::endl;

		// bubble sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		bubbleSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][0] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "bubble sort time cost: " << total_time[i][0] << std::endl;

		// insertion sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		// insertionSort(sequence, sequence.size());
		insertionSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][1] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "insertion sort time cost: " << total_time[i][1] << std::endl;

		// selection sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		selectionSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][2] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "selection sort time cost: " << total_time[i][2] << std::endl;

		// merge sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		mergeSort(sequence, 0, sequence.size());
		end = clock();
		sequence.clear();
		total_time[i][3] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "merge sort time cost: " << total_time[i][3] << std::endl;

		// quick sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		quickSort(sequence, 0, sequence.size());
		end = clock();
		sequence.clear();
		total_time[i][4] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "quick sort time cost: " << total_time[i][4] << std::endl;
		
		// heap sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		heapSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][5] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "heap sort time cost: " << total_time[i][5] << std::endl;

		// shell sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		shellSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][6] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "shell sort time cost: " << total_time[i][6] << std::endl;

		// counting sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		countingSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][7] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "counting sort time cost: " << total_time[i][7] << std::endl;

		// bucket sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		int bucket_count = 50;
		bucketSort(sequence, bucket_count);
		end = clock();
		sequence.clear();
		total_time[i][8] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "bucket sort time cost: " << total_time[i][8] << std::endl;

		// radix sort
		sequence.assign(seq.begin(), seq.end());
		start = clock();
		radixSort(sequence);
		end = clock();
		sequence.clear();
		total_time[i][9] = (double)(end - start) / (double)(CLOCKS_PER_SEC);
		std::cout << "radix sort time cost: " << total_time[i][9] << std::endl;

		std::cout << std::endl << std::endl << std::endl;
	}
	return 1;
}

/*
int main()
{
	int n = 1000;
	std::vector<int> seq;
	seqGenerate(seq, n);
	radixSort(seq);
	for (int i = 0; i < seq.size(); i++)
	{
		std::cout << seq[i] << "  ";
	}
	std::cout << std::endl;
	int flag = 0;
	for (int i = 0; i < seq.size() - 1; i++)
	{
		if (seq[i] > seq[i + 1])
		{
			flag = 1;
			break;
		}
	}
	if (flag == 1)	std::cout << "false!" << std::endl;
	return 1;
}
*/

void seqGenerate(std::vector<int>& seq, int n)
{
	srand((int)time(NULL));
	for (int i = 0; i < n; i++)
	{
		int random_number = rand();
		seq.push_back(random_number);
	}
}

void swap(std::vector<int>& seq, int i, int j)
{
	int temp = seq[i];
	seq[i] = seq[j];
	seq[j] = temp;
}

void bubbleSort(std::vector<int>& seq)
{
	int n = seq.size();
	for (int i = 0; i < n - 1; i++)
	{
		for (int j = 0; j < n - i - 1; j++)
		{
			if (seq[j] > seq[j + 1])
			{
				swap(seq, j, j + 1);
			}
		}
	}
	return;
}

void insertionSort(std::vector<int>& seq)
{
	if (seq.size() <= 1)	return;
	int j, start;
	for (int i = 0; i < seq.size() - 1; i++)
	{
		j = i + 1;
		start = i;
		if (seq[j] > seq[start])	continue;
		while (start >= 0 && seq[start] > seq[start + 1])
		{
			swap(seq, start, start + 1);
			start--;
		}
	}
	return;
}

void insertionSort(std::vector<int>& seq, int n)
{
	if (n == 1)	return;
	insertionSort(seq, n - 1);
	int i = n - 2;
	
	/*
	int value = seq[n - 1];
	while (i >= 0 && seq[i] > seq[i + 1])
	{
		seq[i + 1] = seq[i];
		seq[i] = value;
		i--;
	}
	*/

	while (i >= 0 && seq[i] > seq[i + 1])
	{
		swap(seq, i, i + 1);
		i--;
	}

	return;
}

void selectionSort(std::vector<int>& seq)
{
	int n = seq.size();
	int min_idx = 0;
	for (int i = 0; i < n; i++)
	{
		min_idx = i;
		for (int j = i + 1; j < n; j++)
		{
			if (seq[j] < seq[min_idx]) { min_idx = j; }
		}
		if (min_idx != i) { swap(seq, min_idx, i); }
	}
	return;
}

void mergeSort(std::vector<int>& seq, int lo, int hi)
{
	if (hi - lo == 1)	return;
	int mid = (lo + hi) / 2;
	mergeSort(seq, lo, mid);
	mergeSort(seq, mid, hi);
	int* a;
	a = new int[mid - lo];
	for (int i = 0; i < mid - lo; i++)
	{
		a[i] = seq[lo + i];
	}
	int* b;
	b = new int[hi - mid];
	for (int i = 0; i < hi - mid; i++)
	{
		b[i] = seq[mid + i];
	}
	int j = 0;
	int k = 0;
	int l = lo;
	while (j < mid - lo && k < hi - mid)
	{
		if (a[j] < b[k])
		{
			seq[l] = a[j];
			l++;
			j++;
		}
		else
		{
			seq[l] = b[k];
			l++;
			k++;
		}
	}
	if (j == mid - lo)
	{
		while (k < hi - mid)
		{
			seq[l] = b[k];
			l++;
			k++;
		}
	}
	else if (k == hi - mid)
	{
		while (j < mid - lo)
		{
			seq[l] = a[j];
			l++;
			j++;
		}
	}
	return;
}

void quickSort(std::vector<int>& seq, int lo, int hi)
{
	if (hi - lo == 1 || lo >= hi)	return;
	int partition_pos = partition(seq, lo, hi);
	quickSort(seq, lo, partition_pos);
	quickSort(seq, partition_pos + 1, hi);
}

int partition(std::vector<int>& seq, int lo, int hi) {
	int value = seq[lo];
	int i, j;
	i = lo + 1; 
	j = hi - 1;
	while (i <= j) 
	{
		if (seq[i] > value) 
		{
			swap(seq, i, j);
			j--;
		}
		else
		{
			i++;
		}
	}
	swap(seq, lo, j);
	return j;
}

void heapSort(std::vector<int>& seq)
{
	int n = seq.size();
	for (int i = n / 2 - 1; i >= 0; i--)
	{
		adjustHeap(seq, i, n);
	}
	for (int i = n - 1; i > 0; i--)
	{
		swap(seq, 0, i);
		adjustHeap(seq, 0, i);
	}
	return;
}

void adjustHeap(std::vector<int>& seq, int i, int n)
{
	int l, r, max_idx;
	while (1)
	{
		max_idx = i;
		l = 2 * i + 1;
		r = 2 * i + 2;
		if (l < n && seq[l] > seq[max_idx])
		{
			max_idx = l;
		}
		if (r < n && seq[r] > seq[max_idx])
		{
			max_idx = r;
		}
		if (max_idx != i) 
		{
			swap(seq, max_idx, i); 
			i = max_idx;
		}
		else
		{
			break;
		}
	}
}

void shellSort(std::vector<int>& seq)
{
	if (seq.size() <= 1)	return;
	int gap = seq.size() / 2;
	while (gap > 0)
	{
		for (int i = gap; i < seq.size(); i++)
		{
			int j = i - gap;
			while (j >= 0 && seq[j] > seq[j + gap])
			{
				swap(seq, j, j + gap);
				j -= gap;
			}
		}
		gap /= 2;
	}
	return;
}

void countingSort(std::vector<int>& seq)
{
	if (seq.size() <= 1)	return;
	int seq_min = INT_MAX;
	int seq_max = INT_MIN;
	for (int i = 0; i < seq.size(); i++)
	{
		if (seq[i] < seq_min)	seq_min = seq[i];
		if (seq[i] > seq_max)	seq_max = seq[i];
	}
	std::vector<int> counting(seq_max - seq_min + 1, 0);
	for (int i = 0; i < seq.size(); i++)
	{
		counting[seq[i] - seq_min]++;
	}
	int i = 0;
	int index = 0;
	while (i < seq_max - seq_min + 1)
	{
		if (counting[i] != 0)
		{
			seq[index] = i + seq_min;
			index++;
			counting[i]--;
		}
		else
			i++;
	}
	return;
}

void bucketSort(std::vector<int>& seq, int bucket_count)
{
	if (seq.size() <= 1)	return;
	int seq_min = INT_MAX;
	int seq_max = INT_MIN;
	for (int i = 0; i < seq.size(); i++)
	{
		if (seq[i] < seq_min) { seq_min = seq[i]; }
		if (seq[i] > seq_max) { seq_max = seq[i]; }
	}
	std::vector<std::vector<int> > buckets(bucket_count + 1, std::vector<int>());
	for (int i = 0; i < seq.size(); i++)
	{
		 int bucket_index = int(double(seq[i] - seq_min) / double(seq_max - seq_min + 1) * bucket_count);
		appendElement(buckets[bucket_index], seq[i]);
	}
	int k = 0;
	for (int i = 0; i < bucket_count; i++)
	{
		for (int j = 0; j < buckets[i].size(); j++)
		{
			seq[k] = buckets[i][j];
			k++;
		}
	}
	return;
}

void appendElement(std::vector<int>& bucket, int val)
{
	bucket.push_back(val);
	for (int i = bucket.size() - 2; i >= 0; i--)
	{
		if (bucket[i] > bucket[i + 1]) { swap(bucket, i, i + 1); }
	}
}

void radixSort(std::vector<int>& seq)
{
	if (seq.size() <= 1)	return;
	int seq_max = INT_MIN;
	for (int i = 0; i < seq.size(); i++)
	{
		if (seq[i] > seq_max) { seq_max = seq[i]; }
	}
	int bit_num = 0;
	while (seq_max > 0)
	{
		seq_max /= 10;
		bit_num++;
	}
	int k = 0;
	int bit = 0;
	int n = 1;
	std::vector<std::vector<int> > bit_array(10, std::vector<int>());
	while (bit < bit_num)
	{
		for (int i = 0; i < seq.size(); i++)
		{
			int bit_idx = (seq[i] / n) % 10;
			bit_array[bit_idx].push_back(seq[i]);
		}
		for (int i = 0; i < 10; i++)
		{
			if (!bit_array[i].empty())
			{
				for (int j = 0; j < bit_array[i].size(); j++)
				{
					seq[k] = bit_array[i][j];
					k++;
				}
			}
			bit_array[i].clear();
		}
		n *= 10;
		k = 0;
		bit++;
	}
	return;
}
