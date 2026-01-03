"""
GPU Job Scheduling Optimization
Compares FCFS vs Priority-Weighted Greedy scheduling
"""

import random
import json
from typing import List, Dict, Tuple

def generate_jobs(n: int = 50) -> List[Dict]:
    """Generate synthetic ML training jobs with varying characteristics"""
    jobs = []
    for i in range(n):
        jobs.append({
            'id': i,
            'priority': random.randint(1, 10),  # 10 = critical, 1 = low
            'gpus': random.choice([1, 2, 4, 8]),
            'duration': random.randint(1, 12),  # hours
            'deadline': random.randint(24, 168),  # hours from submission
            'arrival_time': random.randint(0, 24)  # when job was submitted
        })
    return jobs

def fcfs_schedule(jobs: List[Dict]) -> List[Tuple[Dict, int]]:
    """First-Come-First-Served scheduling (baseline)"""
    # Sort by arrival time
    sorted_jobs = sorted(jobs, key=lambda j: j['arrival_time'])
    
    schedule = []
    current_time = 0
    
    for job in sorted_jobs:
        # Start job as soon as it arrives or previous job finishes
        start_time = max(job['arrival_time'], current_time)
        schedule.append((job, start_time))
        current_time = start_time + job['duration']
    
    return schedule

def greedy_schedule(jobs: List[Dict]) -> List[Tuple[Dict, int]]:
    """Priority-weighted greedy scheduling"""
    # Sort by priority per resource-hour (higher = schedule first)
    sorted_jobs = sorted(jobs, 
                        key=lambda j: j['priority'] / (j['gpus'] * j['duration']),
                        reverse=True)
    
    schedule = []
    current_time = 0
    
    for job in sorted_jobs:
        # Start job as soon as it arrives or previous job finishes
        start_time = max(job['arrival_time'], current_time)
        schedule.append((job, start_time))
        current_time = start_time + job['duration']
    
    return schedule

def calculate_metrics(schedule: List[Tuple[Dict, int]], 
                     gpu_capacity: int = 32) -> Dict:
    """Calculate scheduling performance metrics"""
    if not schedule:
        return {'utilization': 0, 'deadline_miss_rate': 0, 'avg_latency': 0}
    
    total_time = max(start + job['duration'] for job, start in schedule)
    
    # GPU utilization over time
    gpu_usage = [0] * (total_time + 1)
    for job, start in schedule:
        for t in range(start, min(start + job['duration'], total_time + 1)):
            gpu_usage[t] += job['gpus']
    
    utilization = sum(gpu_usage) / (gpu_capacity * total_time) * 100
    
    # Deadline misses
    misses = sum(1 for job, start in schedule 
                 if start + job['duration'] > job['deadline'])
    deadline_miss_rate = (misses / len(schedule)) * 100
    
    # Average job latency (time from submission to start)
    latencies = [start - job['arrival_time'] for job, start in schedule]
    avg_latency = sum(latencies) / len(latencies)
    
    return {
        'utilization': round(utilization, 1),
        'deadline_miss_rate': round(deadline_miss_rate, 1),
        'avg_latency': round(avg_latency, 2)
    }

def run_comparison(n_jobs: int = 50, seed: int = 42):
    """Run scheduling comparison and print results"""
    random.seed(seed)
    jobs = generate_jobs(n_jobs)
    
    print(f"\n{'='*60}")
    print(f"GPU Job Scheduling Optimization")
    print(f"{'='*60}\n")
    print(f"Simulating {n_jobs} jobs on 32-GPU cluster\n")
    
    # FCFS baseline
    fcfs_sched = fcfs_schedule(jobs)
    fcfs_metrics = calculate_metrics(fcfs_sched)
    
    # Greedy scheduler
    greedy_sched = greedy_schedule(jobs)
    greedy_metrics = calculate_metrics(greedy_sched)
    
    # Print comparison
    print(f"{'Metric':<25} {'FCFS':<15} {'Greedy':<15} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    metrics = ['utilization', 'deadline_miss_rate', 'avg_latency']
    labels = ['GPU Utilization (%)', 'Deadline Miss Rate (%)', 'Avg Job Latency (hrs)']
    
    for metric, label in zip(metrics, labels):
        fcfs_val = fcfs_metrics[metric]
        greedy_val = greedy_metrics[metric]
        
        if metric == 'utilization':
            improvement = f"+{greedy_val - fcfs_val:.1f}%"
        else:
            pct_change = ((fcfs_val - greedy_val) / fcfs_val * 100) if fcfs_val > 0 else 0
            improvement = f"{pct_change:+.1f}%"
        
        print(f"{label:<25} {fcfs_val:<15} {greedy_val:<15} {improvement:<15}")
    
    print(f"\n{'='*60}\n")
    
    return {
        'fcfs': fcfs_metrics,
        'greedy': greedy_metrics,
        'jobs': jobs
    }

if __name__ == "__main__":
    results = run_comparison(n_jobs=100)