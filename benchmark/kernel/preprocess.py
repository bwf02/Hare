import re
import logging
import subprocess
from datetime import datetime

def setup_logging(filename=None, level=logging.INFO):
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | » %(message)s', 
        datefmt='%m-%d %H:%M'
    )

    if filename is None:
        filename = datetime.now().strftime("log_%Y%m%d_%H%M%S.log")

    root = logging.getLogger()
    root.setLevel(level)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # 日志写入文件
    data_logger = logging.getLogger("file")
    data_logger.propagate = False
    
    fh = logging.FileHandler(filename, mode='w')
    fh.setFormatter(formatter)
    data_logger.addHandler(fh)


def parse_nsys_avg_ns(text: str, kernel_name: str):
    pattern = re.compile(
        r"\s*[\d\.]+\s+"     # Time (%)
        r"(\d+)\s+"          # Total Time (ns)
        r"(\d+)\s+"          # Instances
        r"([\d\.]+)\s+"      # Avg (ns)  <-- 我们要的
        r"[\d\.]+\s+"        # Med
        r"\d+\s+"            # Min
        r"\d+\s+"            # Max
        r"[\d\.]+\s+"        # StdDev
        r"(.+)$"             # Name
    )
    result = 0
    for line in text.splitlines():
        line = line.strip()
        if kernel_name in line:
            m = pattern.match(line)
            if not m:
                continue
            avg_ns = float(m.group(3))
            result = avg_ns
            break

    return result

if __name__ == "__main__":
    
    log_str = """
WARNING: python and any of its children processes will be profiled.

Collecting data...
/root/code/fast-sparse/third_party/megablocks/megablocks/grouped_gemm_util.py:10: UserWarning: Grouped GEMM not available.
  warnings.warn('Grouped GEMM not available.')
INFO 12-08 00:54:14 [__init__.py:216] Automatically detected platform cuda.
WARNING 12-08 00:54:14 [cuda.py:682] Detected different devices in the system: NVIDIA GeForce RTX 5080, NVIDIA GeForce RTX 3090. Please make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to avoid unexpected behavior.
Avg kernel time = 0.121235 ms
Generating '/tmp/nsys-report-e7dc.qdstrm'
[1/7] [========================100%] test.nsys-rep
[2/7] [========================100%] test.sqlite
[3/7] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)   Style                       Range                    
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -------  ---------------------------------------------
     54.5           149854          2   74927.0   74927.0     31351    118503      61625.8  PushPop  CCCL:cub::DeviceScan::InclusiveSum           
     25.5            70053          1   70053.0   70053.0     70053     70053          0.0  PushPop  CCCL:cub::DeviceRadixSort                    
     20.1            55282          1   55282.0   55282.0     55282     55282          0.0  PushPop  CCCL:cub::DeviceHistogram::MultiHistogramEven

[4/7] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                Name               
 --------  ---------------  ---------  ---------  ----------  --------  --------  -----------  ---------------------------------
     48.2        730849179     133125     5489.9      5272.0      4965   2539306      10799.3  cudaMemcpyAsync                  
     29.3        445012274     133124     3342.8      3368.0      1710    117284        807.2  cudaStreamSynchronize            
     18.9        286434502        136  2106136.0     19845.0      7462  45284717    7777570.5  cudaLaunchKernel                 
      2.6         39343912          4  9835978.0  11229815.5   5129300  11754981    3151973.9  cuLibraryLoadData                
      0.3          3979776          2  1989888.0   1989888.0   1012929   2966847    1381628.7  cudaGetDeviceProperties_v2_v12000
      0.3          3812561          1  3812561.0   3812561.0   3812561   3812561          0.0  cudaFree                         
      0.2          2342146        111    21100.4      1465.0       904   2199134     208609.3  cuKernelGetFunction              
      0.1          2043242        110    18574.9     19584.0     13296     47668       4942.1  cudaLaunchKernelExC_v11060       
      0.1          1449183          1  1449183.0   1449183.0   1449183   1449183          0.0  cudaHostAlloc                    
      0.1           853593          6   142265.5    135160.0      7948    301702     100376.2  cudaMalloc                       
      0.0           725309          1   725309.0    725309.0    725309    725309          0.0  cudaLibraryLoadFromFile_v12060   
      0.0           223733        838      267.0       241.5       126      3154        162.4  cuGetProcAddress_v2              
      0.0           149067        220      677.6       598.0       300      2076        253.8  cuTensorMapEncodeTiled           
      0.0            49880          2    24940.0     24940.0      8824     41056      22791.5  cudaDeviceSynchronize            
      0.0            42485         20     2124.3       491.5       453     14139       3812.3  cudaEventCreateWithFlags         
      0.0            40712          2    20356.0     20356.0     11844     28868      12037.8  cudaEventRecordWithFlags_v11010  
      0.0            35395        110      321.8       360.0       224       598         71.2  cuFuncGetName                    
      0.0            33780          9     3753.3      2727.0       452     15230       4492.3  cudaStreamIsCapturing_v10000     
      0.0            31120          1    31120.0     31120.0     31120     31120          0.0  cuLaunchKernel                   
      0.0            22925          5     4585.0      5141.0       862      8296       3443.1  cuLibraryGetKernel               
      0.0            14896          4     3724.0      3119.5      2099      6558       1990.3  cuInit                           
      0.0             6295          2     3147.5      3147.5      1414      4881       2451.5  cudaEventQuery                   
      0.0             4109          2     2054.5      2054.5      1202      2907       1205.6  cudaEventDestroy                 
      0.0             3428          1     3428.0      3428.0      3428      3428          0.0  cudaLibraryGetKernel_v12060      
      0.0             2011          1     2011.0      2011.0      2011      2011          0.0  cudaLibraryUnload_v12060         
      0.0             1867          2      933.5       933.5       492      1375        624.4  cudaGetDriverEntryPoint_v11030   
      0.0             1551          3      517.0       202.0       180      1169        564.8  cuModuleGetLoadingMode           

[5/7] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     49.5           259105        110    2355.5    2368.0      2336      2400         18.9  void deep_gemm::fastdss_impl<(int)1024, (int)1024, (int)8192, (int)16, (int)10, (int)4, (int)1, (in…
     38.3           200704        110    1824.6    1824.0      1792      1856         20.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::BFloat16>, std:…
      2.1            10976          1   10976.0   10976.0     10976     10976          0.0  void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_tn_align1>(T1::Params)                       
      1.8             9504          1    9504.0    9504.0      9504      9504          0.0  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<float, (int)4, void at::nat…
      1.0             5152          1    5152.0    5152.0      5152      5152          0.0  void at::native::sbtopk::gatherTopK<float, unsigned int, (int)2, (bool)0>(at::cuda::detail::TensorI…
      0.9             4960          2    2480.0    2480.0      2432      2528         67.9  void megablocks::cub::DeviceScanKernel<megablocks::cub::detail::scan::policy_hub<int, int, int, uns…
      0.9             4480          1    4480.0    4480.0      4480      4480          0.0  void megablocks::cub::DeviceHistogramSweepKernel<megablocks::cub::detail::histogram::policy_hub<int…
      0.7             3584          1    3584.0    3584.0      3584      3584          0.0  void megablocks::cub::DeviceRadixSortSingleTileKernel<megablocks::cub::detail::radix::policy_hub<in…
      0.6             2880          3     960.0     928.0       928      1024         55.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<int>, std::array<cha…
      0.5             2848          1    2848.0    2848.0      2848      2848          0.0  void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapp…
      0.4             2240          2    1120.0    1120.0      1088      1152         45.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<unsigned int>, std::…
      0.4             1856          2     928.0     928.0       928       928          0.0  void megablocks::cub::DeviceScanInitKernel<megablocks::cub::ScanTileState<int, (bool)1>>(T1, int)   
      0.3             1728          1    1728.0    1728.0      1728      1728          0.0  void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIterator…
      0.3             1440          1    1440.0    1440.0      1440      1440          0.0  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)0, floa…
      0.3             1408          1    1408.0    1408.0      1408      1408          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<int, int, int, at:…
      0.3             1408          1    1408.0    1408.0      1408      1408          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      0.3             1376          1    1376.0    1376.0      1376      1376          0.0  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<float, (int)4, void at::nat…
      0.2             1248          1    1248.0    1248.0      1248      1248          0.0  void <unnamed>::softmax_warp_forward<float, float, float, (int)2, (bool)0, (bool)0>(T2 *, const T1 …
      0.2             1216          1    1216.0    1216.0      1216      1216          0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<int>, std:…
      0.2             1216          1    1216.0    1216.0      1216      1216          0.0  void getColIndicesKernel<int>(T1 *, T1 *, int, int, int, int)                                       
      0.2             1216          1    1216.0    1216.0      1216      1216          0.0  void getHeightOffsetKernel<int>(T1 *, T1 *, int, int)                                               
      0.2             1056          1    1056.0    1056.0      1056      1056          0.0  void getRowIndicesKernel<int>(T1 *, T1 *, T1 *, int, int, int)                                      
      0.2              896          1     896.0     896.0       896       896          0.0  void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &,…
      0.2              896          1     896.0     896.0       896       896          0.0  void megablocks::cub::DeviceHistogramInitKernel<(int)1, int, int>(cuda::std::__4::array<int, T1>, c…

[6/7] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  ------  --------  --------  --------  --------  -----------  ------------------------------
    100.0         41560680  133122     312.2     288.0       256   1557477       4720.7  [CUDA memcpy Host-to-Device]  
      0.0             1280       2     640.0     640.0       352       928        407.3  [CUDA memcpy Device-to-Host]  
      0.0             1120       1    1120.0    1120.0      1120      1120          0.0  [CUDA memcpy Device-to-Device]

[7/7] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 ----------  ------  --------  --------  --------  --------  -----------  ------------------------------
     13.115  133122     0.000     0.000     0.000     8.389        0.026  [CUDA memcpy Host-to-Device]  
      0.004       1     0.004     0.004     0.004     0.004        0.000  [CUDA memcpy Device-to-Device]
      0.000       2     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]  

Generated:
        /tmp/test.nsys-rep
        /tmp/test.sqlite
"""

    avg_ns = parse_nsys_avg_ns(log_str, "fastdss_impl")
    print(avg_ns)