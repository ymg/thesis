#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int data[1000001];
    int largest, largest_factor = 0;

    for (int i_ = 1000; i_ <= 1000000; i_++)
    {
        data[i_] = i_;
    }

    int n_threads = atoi(argv[1]);

    omp_set_num_threads(n_threads);

/* "omp parallel for" turns the for loop multithreaded by making each thread
 * iterating only a part of the loop variable, in this case i; variables declared
 * as "shared" will be implicitly locked on access
 */
#pragma omp parallel for shared(largest_factor, largest)
    for (int i = 1000; i < 1000000; i++)
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

    printf("Largest factor: %d of %d\n", largest_factor, largest);
    return 0;
}