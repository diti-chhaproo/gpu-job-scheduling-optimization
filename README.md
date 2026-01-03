GPU Job Scheduling Optimization:
Optimizing GPU cluster utilization for ML training workloads through priority-weighted scheduling.

Problem:
A shared 32-GPU cluster experienced 62% utilization, 15% deadline misses, and unfair resource allocation under first-come-first-served scheduling.

Solution:
Developed priority-weighted greedy scheduler that ranks jobs by priority per resource-hour ratio. Achieves near-optimal performance in under 100ms vs 2+ minutes for exact optimization.

Results:
GPU Utilization: 62% → 81% (+31%)
Deadline Miss Rate: 15% → 3% (-80%)
Average Job Latency: 4.2hrs → 2.8hrs (-34%)

Usage:
python scheduler.py

Technical Stack:
Python 3.8+ (no external dependencies)
O(n log n) scheduling complexity
Discrete-event simulation for validation

Author:
Diti Chhaproo
University of Illinois Urbana-Champaign
ditichhaproo@gmail.com