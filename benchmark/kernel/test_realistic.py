import csv
import os
import logging

NUM_EXPERT = 4
BW_LIST = [16, 32, 64, 128]

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

# m k n
SHAPES = [
    [16384 * NUM_EXPERT, 6144, 4096],
    [14336 * NUM_EXPERT, 4096, 4096],   
    [5760 * NUM_EXPERT, 2304, 4096],
    [1408 * NUM_EXPERT, 2048, 4096],
    [768 * NUM_EXPERT, 2048, 4096],
]

SPARSITYS = [0.5, 0.75, 0.9]

def run_benchmark():
    output = "result/realistic_5090.csv"
    logfile = "log/realistic_5090.log"
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
    
    for sparsity in SPARSITYS:
        for shape in SHAPES:
            (m, k, n) = shape
            cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res = 0, 0, 0, 0, 0, 0, 0
            ssd_res = test_SSD(m, n, k, sparsity, NUM_EXPERT, bw=BW_LIST[2])
            dss_res = test_DSS(m, n, k, sparsity, NUM_EXPERT, bw=BW_LIST[2])
            sputnik_res = test_Sputnik(m, n, k, sparsity)
            samoyeds_res = test_Samoyeds(m, n, k, sparsity)
            clasp_res = test_CLASP(m, n, k, sparsity)
            spatha_res = test_Spatha(m, n, k, sparsity)
            cublas_res = test_cuBLAS(m, n, k, sparsity)

            csv_row = [m, n, k, sparsity, cublas_res, sputnik_res, samoyeds_res, clasp_res, spatha_res, ssd_res, dss_res]
            logging.info(",".join([str(i) for i in csv_row]))
            writer.writerow(csv_row)
            csvfile.flush()

    csvfile.close()

if __name__ == "__main__":
    run_benchmark()
