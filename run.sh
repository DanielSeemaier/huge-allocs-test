T=${1:-8}
g++ a.cc -O3 -fopenmp && OMP_PROC_BIND="spread" OMP_NUM_THREADS=$T ./a.out

