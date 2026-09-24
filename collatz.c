#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define STUDENT_ID 230103360ULL
#define N 13360000ULL
#define MOD 1000000007ULL
#define MAX_THREADS 32

static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

int main() {
    printf("=== ZEBA ACADEMY LAB: AMDAHL REALITY GAP ===\n");
    printf("Student ID: %llu | Workload N: %llu\n", STUDENT_ID, N);
    printf("Max OpenMP Threads: %d\n\n", omp_get_max_threads());

    // PHASE 2: SEQUENTIAL BASELINE
    printf("--- PHASE 2: Sequential Baseline ---\n");
    uint32_t max_steps_seq = 0;
    uint64_t checksum_seq = 0;

    double t_seq_runs[3];
    for (int run = 0; run < 3; run++) {
        double start = omp_get_wtime();
        uint32_t m_steps = 0;
        uint64_t c_sum = 0;
        
        for (uint64_t i = 1; i <= N; i++) {
            uint32_t s = collatz_steps(i);
            if (s > m_steps) m_steps = s;
            c_sum = (c_sum + s) % MOD;
        }
        
        double end = omp_get_wtime();
        t_seq_runs[run] = end - start;
        max_steps_seq = m_steps;
        checksum_seq = c_sum;
        printf("Run %d: %.6f s | Checksum: %llu\n", run + 1, t_seq_runs[run], (unsigned long long)checksum_seq);
    }
    double T_seq = (t_seq_runs[1] + t_seq_runs[2]) / 2.0;
    printf("Avg T_seq (Runs 2 & 3): %.6f s | Max Steps: %u\n\n", T_seq, max_steps_seq);

    // PHASE 3: MULTI-THREAD SCALING
    printf("--- PHASE 3: Multi-Thread Scaling ---\n");
    int threads_to_test[] = {1, 2, 4, 8, omp_get_max_threads()};
    int num_tests = sizeof(threads_to_test) / sizeof(threads_to_test[0]);

    printf("Threads\tRun 1 (Cold)\tRun 2\t\tRun 3\t\tAvg T_k (s)\tS_emp(k)\n");
    
    for (int idx = 0; idx < num_tests; idx++) {
        int k = threads_to_test[idx];
        double t_runs[3];

        for (int run = 0; run < 3; run++) {
            omp_set_num_threads(k);
            double start = omp_get_wtime();
            
            uint32_t m_steps = 0;
            uint64_t c_sum = 0;

            #pragma omp parallel for reduction(max:m_steps) reduction(+:c_sum)
            for (uint64_t i = 1; i <= N; i++) {
                uint32_t s = collatz_steps(i);
                if (s > m_steps) m_steps = s;
                c_sum += s;
            }

            double end = omp_get_wtime();
            t_runs[run] = end - start;
        }

        double Avg_Tk = (t_runs[1] + t_runs[2]) / 2.0;
        double S_emp = T_seq / Avg_Tk;

        printf("%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.4f\n", k, t_runs[0], t_runs[1], t_runs[2], Avg_Tk, S_emp);
    }

    // PHASE 4: EXPERIMENT A (FALSE SHARING)
    printf("\n--- PHASE 4: Experiment A (False Sharing) ---\n");
    int max_k = omp_get_max_threads();
    omp_set_num_threads(max_k);

    uint64_t naive_hits[MAX_THREADS] = {0};
    double t_start = omp_get_wtime();
    #pragma omp parallel for
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > 100) {
            naive_hits[omp_get_thread_num()]++;
        }
    }
    double t_v1 = omp_get_wtime() - t_start;

    uint64_t total_hits_red = 0;
    t_start = omp_get_wtime();
    #pragma omp parallel for reduction(+:total_hits_red)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > 100) {
            total_hits_red++;
        }
    }
    double t_v2 = omp_get_wtime() - t_start;

    printf("Variant 1 (Naive array - False Sharing): Time = %.6f s\n", t_v1);
    printf("Variant 2 (Reduction - Mitigated)    : Time = %.6f s\n", t_v2);
    printf("Slowdown Ratio (V1 / V2)               : %.2fx\n", t_v1 / t_v2);

    // PHASE 4: EXPERIMENT B (SCHEDULING)
    printf("\n--- PHASE 4: Experiment B (Scheduling) ---\n");
    
    t_start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 1; i <= N; i++) { collatz_steps(i); }
    printf("schedule(static)         : %.6f s\n", omp_get_wtime() - t_start);

    t_start = omp_get_wtime();
    #pragma omp parallel for schedule(static, 1000)
    for (uint64_t i = 1; i <= N; i++) { collatz_steps(i); }
    printf("schedule(static, 1000)   : %.6f s\n", omp_get_wtime() - t_start);

    t_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 100)
    for (uint64_t i = 1; i <= N; i++) { collatz_steps(i); }
    printf("schedule(dynamic, 100)   : %.6f s\n", omp_get_wtime() - t_start);

    t_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 10000)
    for (uint64_t i = 1; i <= N; i++) { collatz_steps(i); }
    printf("schedule(dynamic, 10000) : %.6f s\n", omp_get_wtime() - t_start);

    t_start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (uint64_t i = 1; i <= N; i++) { collatz_steps(i); }
    printf("schedule(guided)        : %.6f s\n", omp_get_wtime() - t_start);

    return 0;
}