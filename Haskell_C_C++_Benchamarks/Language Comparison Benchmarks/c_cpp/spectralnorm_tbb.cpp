// The Computer Language Benchmarks Game
// https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
//
// Original C contributed by Sebastien Loisel
// Conversion to C++ by Jon Harrop
// OpenMP parallelize by The Anh Tran
// Add SSE by The Anh Tran

// Fastest with this flag: -Os
// g++ -pipe -Os -fomit-frame-pointer -march=native -fopenmp -mfpmath=sse -msse2 ./spec.c++ -o ./spec.run

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include "tbb/tbb.h"
#include "tbb/pipeline.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

using namespace std;

double eval_A(int i, int j) { return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1); }

void eval_A_times_u(const vector<double> &u, vector<double> &Au)
{
    for (int i = 0; i < u.size(); i++)
        for (int j = 0; j < u.size(); j++)
            Au[i] += eval_A(i, j) * u[j];
}

void eval_At_times_u(const vector<double> &u, vector<double> &Au)
{
    for (int i = 0; i < u.size(); i++)
        for (int j = 0; j < u.size(); j++)
            Au[i] += eval_A(j, i) * u[j];
}

void eval_AtA_times_u(const vector<double> &u, vector<double> &AtAu)
{
    vector<double> v(u.size());
    eval_A_times_u(u, v);
    eval_At_times_u(v, AtAu);
}

int main(int argc, char *argv[])
{
    int N = ((argc == 2) ? atoi(argv[1]) : 2000);
    int nthreads = atoi(argv[2]);

    vector<double> u(N), v(N);

    fill(u.begin(), u.end(), 1);

    tbb::task_scheduler_init init(nthreads);

    static tbb::spin_mutex mtx;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, 10),
                      [&u, &v, &N](const tbb::blocked_range<size_t> &r)
                      {
                          tbb::spin_mutex::scoped_lock lock(mtx);
                          {
                              for (int i = r.begin(); i < r.end(); i++)
                              {

                                  eval_AtA_times_u(u, v);

                                  fill(u.begin(), u.end(), 0);

                                  eval_AtA_times_u(v, u);
                              }
                          }
                      });

    double vBv = 0, vv = 0;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, N),
                      [&u, &v, &N, &vv, &vBv](const tbb::blocked_range<size_t> &r)
                      {
                          tbb::spin_mutex::scoped_lock lock(mtx);
                          {
                              for (int i = 0; i < N; i++)
                              {
                                  vBv += u[i] * v[i];
                                  vv += v[i] * v[i];
                              }
                          }
                      });

    cout << setprecision(10) << sqrt(vBv / vv) << endl;

    return 0;
}
