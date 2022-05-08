#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <pthread.h>

#define ITERATION 10

using namespace std;

struct thread_data
{
  int start;
  int end;
  double vBv;
  double vv;
};

int size;
vector<double> u, v, tmp;

pthread_barrier_t u_v_tmp_barrier[ITERATION];
pthread_barrier_t u_v_end_barrier[ITERATION];
pthread_barrier_t v_u_tmp_barrier[ITERATION];
pthread_barrier_t v_u_end_barrier[ITERATION];

double eval_A(int i, int j)
{
  return 1.0 / ((i + j) * (i + j + 1) / 2 + i + 1);
}

void eval_A_times_u(const vector<double> &u, vector<double> &Au, int start, int end)
{
  for (int i = start; i <= end; i++)
  {
    for (int j = 0; j < u.size(); j++)
    {
      Au[i] += eval_A(i, j) * u[j];
    }
  }
}

void eval_At_times_u(const vector<double> &u, vector<double> &Au, int start, int end)
{
  for (int i = start; i <= end; i++)
  {
    for (int j = 0; j < u.size(); j++)
    {
      Au[i] += eval_A(j, i) * u[j];
    }
  }
}

void eval_AtA_times_u(const vector<double> &u, vector<double> &AtAu, int start, int end, pthread_barrier_t *tmp_barrier, pthread_barrier_t *end_barrier)
{
  fill(tmp.begin() + start, tmp.begin() + 1 + end, 0);
  eval_A_times_u(u, tmp, start, end);

  pthread_barrier_wait(tmp_barrier);
  eval_At_times_u(tmp, AtAu, start, end);

  pthread_barrier_wait(end_barrier);
}

void *spectral_multithread(void *arg)
{
  thread_data *data = (thread_data *)arg;
  for (int i = 0; i < ITERATION; i++)
  {
    eval_AtA_times_u(u, v, data->start, data->end, &(u_v_tmp_barrier[i]), &(u_v_end_barrier[i]));
    fill(u.begin() + data->start, u.begin() + 1 + data->end, 0);
    eval_AtA_times_u(v, u, data->start, data->end, &(v_u_tmp_barrier[i]), &(v_u_end_barrier[i]));
  }
  data->vBv = 0;
  data->vv = 0;
  for (int i = data->start; i <= data->end; i++)
  {
    data->vBv += u[i] * v[i];
    data->vv += v[i] * v[i];
  }
  pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
  size = ((argc == 2) ? atoi(argv[1]) : 2000);

  const int THREAD_COUNT = atoi(argv[2]);

  u.assign(size, 1);
  v.assign(size, 0);
  tmp.assign(size, 0);

  pthread_t thread_ids[THREAD_COUNT];
  thread_data datas[THREAD_COUNT];
  for (int i = 0; i < ITERATION; i++)
  {
    pthread_barrier_init(&(u_v_tmp_barrier[i]), NULL, THREAD_COUNT);
    pthread_barrier_init(&(v_u_tmp_barrier[i]), NULL, THREAD_COUNT);
    pthread_barrier_init(&(u_v_end_barrier[i]), NULL, THREAD_COUNT);
    pthread_barrier_init(&(v_u_end_barrier[i]), NULL, THREAD_COUNT);
  }

  int thread_size = size / THREAD_COUNT;
  int extras = size % THREAD_COUNT;
  for (int i = 0; i < THREAD_COUNT; i++)
  {
    datas[i].start = i * thread_size;
    datas[i].end = datas[i].start + thread_size - 1;
  }
  datas[THREAD_COUNT - 1].end += extras;
  for (int i = 0; i < THREAD_COUNT; i++)
  {
    if (pthread_create(&(thread_ids[i]), NULL, &spectral_multithread, static_cast<void *>(&datas[i])) != 0)
    {
      cout << "ERROR CREATING THREAD " << i << endl;
    }
  }
  double vBv = 0, vv = 0;
  for (int i = 0; i < THREAD_COUNT; i++)
  {
    pthread_join(thread_ids[i], NULL);
    vBv += datas[i].vBv;
    vv += datas[i].vv;
  }
  cout << setprecision(10) << sqrt(vBv / vv) << endl;
  return 0;
}

