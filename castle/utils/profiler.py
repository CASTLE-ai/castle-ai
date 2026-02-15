
import time
from collections import defaultdict
import statistics
import json
import threading

class Profiler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Profiler, cls).__new__(cls)
                cls._instance.timings = defaultdict(list)
        return cls._instance

    def add_record(self, name, duration_ms):
        with self._lock:
            self.timings[name].append(duration_ms)

    def get_report(self):
        report = {}
        with self._lock:
            for name, durations in self.timings.items():
                if not durations:
                    continue
                report[name] = {
                    "count": len(durations),
                    "total_ms": sum(durations),
                    "mean_ms": statistics.mean(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0
                }
        return report

    def print_report(self):
        print("\n" + "="*50)
        print("PERFORMANCE PROFILING REPORT")
        print("="*50)
        report = self.get_report()
        # Sort by total time descending
        sorted_items = sorted(report.items(), key=lambda x: x[1]['total_ms'], reverse=True)
        
        print(f"{'Name':<35} | {'Count':<6} | {'Total(ms)':<10} | {'Mean(ms)':<10} | {'Max(ms)':<10}")
        print("-" * 85)
        for name, stats in sorted_items:
            print(f"{name:<35} | {stats['count']:<6} | {stats['total_ms']:<10.2f} | {stats['mean_ms']:<10.2f} | {stats['max_ms']:<10.2f}")
        print("="*50 + "\n")

    def reset(self):
        with self._lock:
            self.timings.clear()

class TimeBlock:
    def __init__(self, name):
        self.name = name
        self.profiler = Profiler()
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000
        self.profiler.add_record(self.name, duration_ms)

class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.stats = {
            "time": [],
            "cpu_percent": [],
            "gpu_percent": [],
            "gpu_mem_used": []
        }
        self.thread = None

    def _read_cpu_times(self):
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                if line.startswith('cpu '):
                    # cpu  user nice system idle iowait irq softirq steal guest guest_nice
                    parts = line.split()
                    # times: user, nice, system, idle, iowait, irq, softirq, steal
                    times = [float(x) for x in parts[1:8]] 
                    idle = times[3] + times[4]  # idle + iowait
                    active = sum(times) - idle
                    return active, idle + active
        except:
            pass
        return 0, 0

    def _get_gpu_stats(self):
        try:
            import subprocess
            # Query nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=0.1
            )
            if result.returncode == 0:
                # Output format: "util_gpu, mem_used" e.g "30, 4000"
                line = result.stdout.strip().split('\n')[0]
                util_str, mem_str = line.split(',')
                return float(util_str), float(mem_str)
        except:
            pass
        return 0.0, 0.0

    def _monitor_loop(self):
        import time
        last_active, last_total = self._read_cpu_times()
        start_t = time.time()
        
        while self.running:
            time.sleep(self.interval)
            
            # CPU
            curr_active, curr_total = self._read_cpu_times()
            delta_active = curr_active - last_active
            delta_total = curr_total - last_total
            cpu_pct = 0.0
            if delta_total > 0:
                cpu_pct = (delta_active / delta_total) * 100.0
            
            last_active, last_total = curr_active, curr_total
            
            # GPU
            gpu_pct, gpu_mem = self._get_gpu_stats()
            
            # Record
            elapsed = time.time() - start_t
            if self.running:
                with Profiler()._lock:
                    self.stats["time"].append(elapsed)
                    self.stats["cpu_percent"].append(cpu_pct)
                    self.stats["gpu_percent"].append(gpu_pct)
                    self.stats["gpu_mem_used"].append(gpu_mem)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("System Monitoring Started...")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("System Monitoring Stopped.")

    def print_stats(self):
        if not self.stats["cpu_percent"]:
            print("No system stats collected.")
            return

        cpu_avg = statistics.mean(self.stats["cpu_percent"])
        cpu_max = max(self.stats["cpu_percent"])
        
        gpu_avg = statistics.mean(self.stats["gpu_percent"])
        gpu_max = max(self.stats["gpu_percent"])
        
        mem_avg = statistics.mean(self.stats["gpu_mem_used"])
        mem_max = max(self.stats["gpu_mem_used"])
        
        print("\n" + "="*50)
        print("SYSTEM RESOURCE USAGE")
        print("="*50)
        print(f"{'Metric':<20} | {'Average':<10} | {'Max':<10}")
        print("-" * 46)
        print(f"{'CPU Usage (%)':<20} | {cpu_avg:<10.1f} | {cpu_max:<10.1f}")
        print(f"{'GPU Usage (%)':<20} | {gpu_avg:<10.1f} | {gpu_max:<10.1f}")
        print(f"{'GPU Mem (MB)':<20} | {mem_avg:<10.0f} | {mem_max:<10.0f}")
        print("="*50 + "\n")
