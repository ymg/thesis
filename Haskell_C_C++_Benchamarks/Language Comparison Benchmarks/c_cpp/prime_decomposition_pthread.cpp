#include <iostream>
#include <cmath>
#include <pthread.h>

template <typename Fx>
void *start_routine_t(void *arg)
{
    Fx *f = (Fx *)arg;
    return (*f)();
}

int main(int argc, char *argv[])
{

    int data[1000001];
    int largest, largest_factor = 0;

    for (int i_ = 1000; i_ <= 1000000; i_++)
    {
        data[i_] = i_;
    }

    int n_threads = atoi(argv[1]);

    pthread_t nthread[n_threads];

    int split_work_count = std::floor(1000000 / n_threads);

    for (int tid = 1; tid <= n_threads - 1; tid++)
    {
        int start = (tid + 1000);
        int tmp = 100000 - (tid * split_work_count);
        int end = tid * split_work_count + tmp;

        if (tid == n_threads)
        {
            end = 1000000;
        }

        auto search = [&largest_factor, &largest, &start, &end, &data]()
        {
            for (int i = start; i <= end; i++)
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
            };

            return nullptr;
        };

        pthread_create(&nthread[tid], nullptr, start_routine_t<decltype(search)>, &search);
        pthread_join(nthread[tid], nullptr);
    }

    printf("Largest factor: %d of %d\n", largest_factor, largest);
    return 0;
}