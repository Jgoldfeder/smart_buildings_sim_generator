#!/usr/bin/env python3
"""Check status of distributed job."""
import requests
import os

SERVER = os.environ.get("SERVER", "http://128.59.145.47:5000")
API_KEY = os.environ.get("API_KEY", "smart-buildings-key-123")

resp = requests.get(f"{SERVER}/status", headers={"X-API-KEY": API_KEY})
data = resp.json()

total = data["queued"] + data["completed"] + data["failed"] + len(data["active_workers"])
print(f"Completed: {data['completed']}")
print(f"Failed:    {data['failed']}")
print(f"Queued:    {data['queued']}")
print(f"Active:    {len(data['active_workers'])}")
print(f"Total:     {total}")
print()

if data["active_workers"]:
    print("Active workers:")
    for worker, info in data["active_workers"].items():
        pct = (info["step"] / info["total_steps"] * 100) if info["total_steps"] else 0
        print(f"  {worker}: {info['job']} - {info['step']}/{info['total_steps']} ({pct:.1f}%)")
else:
    print("No active workers")

if data["failed_jobs"]:
    print(f"\nRecent failures:")
    for job in data["failed_jobs"]:
        print(f"  {job['job_key']}: {job['error']}")
