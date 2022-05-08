#include <stdio.h>

#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

int main(int argc, char *argv[])
{
    int data[1000001];
    int largest, largest_factor = 0;

    for (int i_ = 1000; i_ <= 1000000; i_++)
    {
        data[i_] = i_;
    }

    int n_threads = atoi(argv[1]);

    tbb::task_scheduler_init init;

    tbb::parallel_for(tbb::blocked_range<size_t>(1000, 1000000),
                      [&largest, &largest_factor, &data](const tbb::blocked_range<size_t> &r)
                      {
                          for (int i = r.begin(); i < r.end(); i++)
                          {
                              int p, n = data[i];

                              for (p = 3; p * p <= n && n % p; p += 2)
                                  ;
                              if (p * p > n)
                                  p = n;

                              if (p > largest_factor)
                              {
                                  largest_factor = p;
                                  largest = n;
                                  // printf("thread %d: found larger: %d of %d\n",
                                  //        omp_get_thread_num(), p, n);
                              }
                          }
                      });

    printf("Largest factor: %d of %d\n", largest_factor, largest);
    return 0;
}