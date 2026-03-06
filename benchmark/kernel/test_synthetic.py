import csv
import os
import logging

from launcher import (
    test_Sputnik,
    test_Samoyeds,
    test_CLASP,
    test_Spatha,
    test_cuBLAS,
    test_SSD,
    test_DSS,
    setup_logging
)

DEFAULT_NUM_EXPERT = 4
VARY_K = [512, 1024, 2048, 4096, 8192, 16384]
VARY_NUM_EXPERT = [2, 4, 6, 8]
VARY_N = [512, 1024, 2048, 4096, 8192, 16384]

# m k n
SHAPES = [
    [16384, 6144, 4096],
    [14336, 4096, 4096],
    [5760, 2304, 4096],
    [1408, 2048, 4096],
    [768, 2048, 4096],
]

SPARSITY = 0.75

def run_benchmark_vary_k():
    output = "result/VaryK.csv"
    logfile = "log/VaryK.log"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    setup_logging(logfile)
    logging.info(f"Logs saved to {logfile}")
    logging.info(f"Results saved to {output}")

    csvfile = open(output, 'w', newline='')
    writer = csv.writer(csvfile)
    header = ['M', "N", "K", "Sparsity", 'cuBLAS','Sputnik', 'Samoyeds', 'CLASP', 'Spatha', 'SSD', 'DSS']
    writer.writerow(header)
    csvfile.flush()
    logging.info(",".join(header))

    for k in VARY_K:
        for shape in SHAPES:
            (m, _, n) = shape
            m = m * DEFAULT_NUM_EXPERT
            cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res = 0, 0, 0, 0, 0, 0, 0
            ssd_res = test_SSD(m, n, k, SPARSITY, DEFAULT_NUM_EXPERT)
            dss_res = test_DSS(m, n, k, SPARSITY, DEFAULT_NUM_EXPERT)
            sputnik_res = test_Sputnik(m, n, k, SPARSITY)
            samoyeds_res = test_Samoyeds(m, n, k, SPARSITY)
            clasp_res = test_CLASP(m, n, k, SPARSITY)
            spatha_res = test_Spatha(m, n, k, SPARSITY)
            cublas_res = test_cuBLAS(m, n, k, SPARSITY)

            csv_row = [m, n, k, SPARSITY, cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res]
            logging.info(csv_row)
            writer.writerow(csv_row)
            csvfile.flush()

    csvfile.close()

def run_benchmark_vary_n():
    output = "result/VaryN.csv"
    logfile = "log/VaryN.log"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    setup_logging(logfile)
    logging.info(f"Logs saved to {logfile}")
    logging.info(f"Results saved to {output}")

    csvfile = open(output, 'w', newline='')
    writer = csv.writer(csvfile)
    header = ['M', "N", "K", "Sparsity", 'cuBLAS','Sputnik', 'Samoyeds', 'CLASP', 'Spatha', 'SSD', 'DSS']
    writer.writerow(header)
    csvfile.flush()
    logging.info(",".join(header))

    for n in VARY_N:
        for shape in SHAPES:
            (m, k, _) = shape
            m = m * DEFAULT_NUM_EXPERT
            cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res = 0, 0, 0, 0, 0, 0, 0
            ssd_res = test_SSD(m, n, k, SPARSITY, DEFAULT_NUM_EXPERT)
            dss_res = test_DSS(m, n, k, SPARSITY, DEFAULT_NUM_EXPERT)
            sputnik_res = test_Sputnik(m, n, k, SPARSITY)
            samoyeds_res = test_Samoyeds(m, n, k, SPARSITY)
            clasp_res = test_CLASP(m, n, k, SPARSITY)
            spatha_res = test_Spatha(m, n, k, SPARSITY)
            cublas_res = test_cuBLAS(m, n, k, SPARSITY)

            csv_row = [m, n, k, SPARSITY, cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res]
            logging.info(csv_row)
            writer.writerow(csv_row)
            csvfile.flush()

    csvfile.close()

def run_benchmark_vary_num_expert():
    output = "result/VaryNumExpert.csv"
    logfile = "log/VaryNumExpert.log"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    setup_logging(logfile)
    logging.info(f"Logs saved to {logfile}")
    logging.info(f"Results saved to {output}")

    csvfile = open(output, 'w', newline='')
    writer = csv.writer(csvfile)
    header = ['M', "N", "K", "NUM_EXPERTS", "Sparsity", 'cuBLAS','Sputnik', 'Samoyeds', 'CLASP', 'Spatha', 'SSD', 'DSS']
    writer.writerow(header)
    csvfile.flush()
    logging.info(",".join(header))

    for num_expert in VARY_NUM_EXPERT:
        for shape in SHAPES:
            (m, k, n) = shape
            m = m * num_expert
            cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res = 0, 0, 0, 0, 0, 0, 0
            ssd_res = test_SSD(m, n, k, SPARSITY, num_expert)
            dss_res = test_DSS(m, n, k, SPARSITY, num_expert)
            sputnik_res = test_Sputnik(m, n, k, SPARSITY)
            samoyeds_res = test_Samoyeds(m, n, k, SPARSITY)
            clasp_res = test_CLASP(m, n, k, SPARSITY)
            spatha_res = test_Spatha(m, n, k, SPARSITY)
            cublas_res = test_cuBLAS(m, n, k, SPARSITY)

            csv_row = [m, n, k, num_expert, SPARSITY, cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res]
            logging.info(",".join([str(i) for i in csv_row]))
            writer.writerow(csv_row)
            csvfile.flush()

    csvfile.close()

if __name__ == "__main__":
    run_benchmark_vary_k()
    run_benchmark_vary_n()
    run_benchmark_vary_num_expert()
