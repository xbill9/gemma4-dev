import pandas as pd
import termplotlib as tpl

# Load the benchmark data
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Error: benchmark_results.csv not found.")
    print("Please run the benchmark suite first.")
    exit()

# Extract data for plotting
concurrency = df["concurrency"].to_numpy()
req_per_sec = df["req_per_sec"].to_numpy()
tokens_per_sec = df["tokens_per_sec"].to_numpy()
avg_latency = df["avg_latency"].to_numpy()

# Create the plots
print("Request Throughput (req/s) vs. Concurrency")
fig1 = tpl.figure()
fig1.plot(concurrency, req_per_sec, label="req/s")
fig1.show()

print("Token Throughput (tok/s) vs. Concurrency")
fig2 = tpl.figure()
fig2.plot(concurrency, tokens_per_sec, label="tok/s")
fig2.show()

print("Average Latency (s) vs. Concurrency")
fig3 = tpl.figure()
fig3.plot(concurrency, avg_latency, label="latency (s)")
fig3.show()
