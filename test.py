import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import LYT
from dataset import create_dataloaders
import os
from torchvision.utils import save_image
import time
import torch.quantization
import copy
from transformer_engine.common.recipe import DelayedScaling, Format
from train import calculate_psnr, calculate_ssim
from tabulate import tabulate
from torch.profiler import profile, record_function, ProfilerActivity

# check for Transformer Engine (for FP8 hardware acceleration)
try:
    import transformer_engine.pytorch as te
    HAS_TRANSFORMER_ENGINE = True
    print("Transformer Engine found - hardware FP8 available")
except ImportError:
    HAS_TRANSFORMER_ENGINE = False

# check for tensorrt (for INT8)
try:
    import tensorrt as trt
    import torch_tensorrt
    HAS_TENSORRT = True
    print("TensorRT found - hardware INT8 available")
except ImportError:
    HAS_TENSORRT = False

def convert_int8(model, sample_inputs, calibration_dataloader=None):
    if not HAS_TENSORRT:
        raise ImportError("TensorRT required for hardware INT8 acceleration")
    
    y, uv = sample_inputs
    
    compiled_model = torch_tensorrt.compile(
        model,
        inputs=[y, uv],
        enabled_precisions={torch.float16, torch.int8},
        workspace_size=1 << 30,
        calibrator=calibration_dataloader if calibration_dataloader else None,
    )
    
    return compiled_model

def optimize_for_inference(model):
    model_opt = torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]])
    return model_opt

def measure_flops(model, inputs, fp16_mode=False):
    """Measure FLOPS for the model with given inputs"""
    if fp16_mode:
        inputs = [x.half() for x in inputs]
    
    # Run warm-up
    model(*inputs)
    torch.cuda.synchronize()
    
    # Run with profiler
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_flops=True, 
                 with_modules=True) as prof:
        with record_function("model_inference"):
            model(*inputs)
    
    torch.cuda.synchronize()
    
    total_flops = 0
    flops_by_operator = {}
    
    for event in prof.key_averages():
        if event.flops > 0:
            total_flops += event.flops
            op_name = event.key if '.' not in event.key else event.key.split('.')[-1]
            if op_name in flops_by_operator:
                flops_by_operator[op_name] += event.flops
            else:
                flops_by_operator[op_name] = event.flops
    
    return total_flops, flops_by_operator

def benchmark_model(model, dataloader, device, num_warmup=10, num_runs=50, fp16_mode=False):
    model.eval()
    
    print("Benchmarking...")
    times = []
    with torch.no_grad():
        for i, (y, uv, _) in enumerate(dataloader):
            if i >= num_runs:
                break
            y, uv = y.to(device), uv.to(device)
            
            if fp16_mode:
                y = y.half()
                uv = uv.half()
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(y, uv)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    times.sort()
    trim_count = len(times) // 10
    if trim_count > 0:
        times = times[trim_count:-trim_count]
    
    avg_time = sum(times) / len(times)
    throughput = 1000 / avg_time 
    
    return {
        "latency_ms": avg_time,
        "throughput": throughput,
        "min_time_ms": min(times),
        "max_time_ms": max(times)
    }

def validate(model, dataloader, device, result_dir, precision="fp32", fp16_mode=False):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    sample_y, sample_uv, _ = next(iter(dataloader))
    sample_y, sample_uv = sample_y.to(device), sample_uv.to(device)
    
    flops, flops_by_op = measure_flops(model, [sample_y, sample_uv], fp16_mode)
    gflops = flops / 1e9
    
    perf_metrics = benchmark_model(model, dataloader, device, fp16_mode=fp16_mode)
    
    flops_per_second = flops / (perf_metrics["latency_ms"] / 1000)
    tflops_per_second = flops_per_second / 1e12
    
    with torch.no_grad():
        for idx, (y, uv, high) in enumerate(dataloader):
            y, uv, high = y.to(device), uv.to(device), high.to(device)
            
            if fp16_mode:
                y = y.half()
                uv = uv.half()
                output = model(y, uv).float() 
            else:
                output = model(y, uv)
                
            output = torch.clamp(output, 0, 1)
            
            save_path = os.path.join(result_dir, f'{precision}_result_{idx}.png')
            save_image(output, save_path)
            
            psnr = calculate_psnr(output, high)
            total_psnr += psnr
            
            ssim = calculate_ssim(output, high)
            total_ssim += ssim
    
    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_allocated = 0
    
    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "latency_ms": perf_metrics["latency_ms"],
        "throughput": perf_metrics["throughput"],
        "min_time_ms": perf_metrics["min_time_ms"],
        "max_time_ms": perf_metrics["max_time_ms"],
        "memory_allocated": memory_allocated,
        "gflops": gflops,
        "tflops_per_second": tflops_per_second,
        "flops_by_operator": flops_by_op
    }

def print_model_results(precision, results, baseline_results=None):
    print(f"\n=== {precision.upper()} Model Results ===")
    
    print("Quality Metrics:")
    print(f"  PSNR: {results['psnr']:.4f}")
    print(f"  SSIM: {results['ssim']:.4f}")
    
    print("Performance Metrics:")
    print(f"  Latency: {results['latency_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput']:.2f} images/sec")
    print(f"  Memory: {results['memory_allocated']:.2f} MB")
    print(f"  GFLOPS: {results['gflops']:.2f}")
    print(f"  TFLOPS/sec: {results['tflops_per_second']:.4f}")
    
    if baseline_results and precision != "fp32":
        print("Comparison to FP32 Baseline:")
        speedup = results['throughput'] / baseline_results['throughput']
        psnr_diff = results['psnr'] - baseline_results['psnr']
        ssim_diff = results['ssim'] - baseline_results['ssim']
        memory_ratio = baseline_results['memory_allocated'] / results['memory_allocated']
        flops_speedup = results['tflops_per_second'] / baseline_results['tflops_per_second']
        
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  FLOPS efficiency gain: {flops_speedup:.2f}x")
        print(f"  PSNR change: {psnr_diff:.4f}")
        print(f"  SSIM change: {ssim_diff:.4f}")
        print(f"  Memory reduction: {memory_ratio:.2f}x")

def print_comparison_table(results):
    print("\n=== Model Precision Comparison ===")
    
    table_data = []
    headers = ["Precision", "PSNR", "SSIM", "Latency (ms)", "Throughput", "Speedup", "TFLOPS/s", "Memory (MB)"]
    
    baseline_throughput = results["fp32"]["throughput"]
    
    for precision, result in results.items():
        speedup = result['throughput'] / baseline_throughput
        
        row = [
            precision.upper(),
            f"{result['psnr']:.4f}",
            f"{result['ssim']:.4f}",
            f"{result['latency_ms']:.2f}",
            f"{result['throughput']:.2f}",
            f"{speedup:.2f}x",
            f"{result['tflops_per_second']:.4f}",
            f"{result['memory_allocated']:.2f}"
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def save_results_to_file(results, dataset_name, cuda_capability):
    results_file = os.path.join('results', dataset_name, 'hardware_quantization_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("=== Hardware-Accelerated Quantization Results ===\n\n")
        
        f.write("System Information:\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}\n")
        f.write(f"CUDA Version: {torch.version.cuda}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Transformer Engine: {'Available' if HAS_TRANSFORMER_ENGINE else 'Not Available'}\n")
        f.write(f"TensorRT: {'Available' if HAS_TENSORRT else 'Not Available'}\n\n")
        
        f.write("Model Precision Comparison:\n")
        
        table_data = []
        headers = ["Precision", "PSNR", "SSIM", "Latency (ms)", "Throughput", "Speedup", "TFLOPS/s", "Memory (MB)"]
        
        baseline_throughput = results["fp32"]["throughput"]
        
        for precision, result in results.items():
            speedup = result['throughput'] / baseline_throughput
            
            row = [
                precision.upper(),
                f"{result['psnr']:.4f}",
                f"{result['ssim']:.4f}",
                f"{result['latency_ms']:.2f}",
                f"{result['throughput']:.2f}",
                f"{speedup:.2f}x",
                f"{result['tflops_per_second']:.4f}",
                f"{result['memory_allocated']:.2f}"
            ]
            table_data.append(row)
        
        f.write(tabulate(table_data, headers=headers, tablefmt="grid") + "\n\n")
        
        f.write("Detailed Results by Precision:\n\n")
        for precision, result in sorted(results.items()):
            f.write(f"{precision.upper()} Results:\n")
            
            f.write("  Quality Metrics:\n")
            f.write(f"    PSNR: {result['psnr']:.4f}\n")
            f.write(f"    SSIM: {result['ssim']:.4f}\n")
            
            f.write("  Performance Metrics:\n")
            f.write(f"    Latency: {result['latency_ms']:.2f} ms\n")
            f.write(f"    Throughput: {result['throughput']:.2f} images/sec\n")
            f.write(f"    Memory usage: {result['memory_allocated']:.2f} MB\n")
            f.write(f"    GFLOPS: {result['gflops']:.2f}\n")
            f.write(f"    TFLOPS/sec: {result['tflops_per_second']:.4f}\n")
            
            f.write("  Top FLOPS by Operator:\n")
            sorted_ops = sorted(result['flops_by_operator'].items(), key=lambda x: x[1], reverse=True)
            for op_name, op_flops in sorted_ops[:5]:  # Show top 5 operators
                op_gflops = op_flops / 1e9
                percentage = (op_flops / (result['gflops'] * 1e9)) * 100
                f.write(f"    {op_name}: {op_gflops:.2f} GFLOPS ({percentage:.1f}%)\n")
            
            if precision != "fp32":
                baseline = results["fp32"]
                f.write("  Comparison to FP32 Baseline:\n")
                speedup = result['throughput'] / baseline['throughput']
                psnr_diff = result['psnr'] - baseline['psnr']
                ssim_diff = result['ssim'] - baseline['ssim']
                memory_ratio = baseline['memory_allocated'] / result['memory_allocated']
                flops_speedup = result['tflops_per_second'] / baseline['tflops_per_second']
                
                f.write(f"    Speedup: {speedup:.2f}x\n")
                f.write(f"    FLOPS efficiency gain: {flops_speedup:.2f}x\n")
                f.write(f"    PSNR change: {psnr_diff:.4f}\n")
                f.write(f"    SSIM change: {ssim_diff:.4f}\n")
                f.write(f"    Memory reduction: {memory_ratio:.2f}x\n")
            
            f.write("\n")
        
    print(f"\nResults saved to {results_file}")

def main():
    test_low = "preprocessed_data/validation/low_yuv/"
    test_high = "preprocessed_data/validation/high_rgb/"
    weights_path = 'models/best_generator.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f"Running on: {torch.cuda.get_device_name(0)}")
        cuda_capability = torch.cuda.get_device_capability(0)
        print(f"CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Running on CPU - hardware acceleration unavailable")
        return

    dataset_name = test_low.split('/')[1]
    
    precisions = ["fp32", "fp16"]
    if HAS_TRANSFORMER_ENGINE:
        precisions.append("fp8")
    if HAS_TENSORRT:
        precisions.append("int8")
    
    result_dirs = {}
    for precision in precisions:
        result_dir = os.path.join('results', dataset_name, precision)
        os.makedirs(result_dir, exist_ok=True)
        result_dirs[precision] = result_dir

    _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=1)
    print(f'Test loader: {len(test_loader)} samples')

    y_sample, uv_sample, _ = next(iter(test_loader))
    y_sample, uv_sample = y_sample.to(device), uv_sample.to(device)
    sample_inputs = (y_sample, uv_sample)

    results = {}
    
    print("\n=== Testing FP32 model (baseline) ===")
    fp32_model = LYT().to(device)
    fp32_model.load_state_dict(torch.load(weights_path, map_location=device))
    
    with torch.no_grad():
        fp32_model(y_sample, uv_sample)
    
    fp32_results = validate(fp32_model, test_loader, device, result_dirs["fp32"], "fp32")
    results["fp32"] = fp32_results
    
    print_model_results("fp32", fp32_results)
    
    print("\n=== Testing FP16 model ===")
    fp16_model = copy.deepcopy(fp32_model).half()
    
    with torch.no_grad():
        y_fp16 = y_sample.half()
        uv_fp16 = uv_sample.half()
        fp16_model(y_fp16, uv_fp16)
    
    fp16_results = validate(fp16_model, test_loader, device, result_dirs["fp16"], "fp16", fp16_mode=True)
    results["fp16"] = fp16_results
    
    print_model_results("fp16", fp16_results, fp32_results)
    
    if HAS_TRANSFORMER_ENGINE:
        print("\n=== Testing FP8 with Transformer Engine ===")
        fp8_model = LYT(filters=32, use_fp8=True).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        fp8_model.load_state_dict(checkpoint, strict=False)
        
        fp8_results = validate(fp8_model, test_loader, device, result_dirs["fp8"], "fp8")
        results["fp8"] = fp8_results
        
        print_model_results("fp8", fp8_results, fp32_results)
    
    if HAS_TENSORRT:
        print("\n=== Testing INT8 model (hardware accelerated) ===")
        base_model = copy.deepcopy(fp32_model)
        
        try:
            int8_model = convert_int8(base_model, sample_inputs)
            
            with torch.no_grad():
                int8_model(y_sample, uv_sample)
            
            int8_results = validate(int8_model, test_loader, device, result_dirs["int8"], "int8")
            results["int8"] = int8_results
            
            print_model_results("int8", int8_results, fp32_results)
        except Exception as e:
            print(f"Error converting to INT8: {e}")
            print("Skipping INT8 evaluation")
    
    print_comparison_table(results)
    save_results_to_file(results, dataset_name, cuda_capability)

if __name__ == '__main__':
    main()