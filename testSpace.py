import subprocess

def test_cpu_performance(num_threads=4, max_prime=20000):
    """
    Runs a CPU benchmark using sysbench to test the processing capabilities of the computer.

    Parameters:
    - num_threads (int): Number of threads to use in the test. Default is 4.
    - max_prime (int): The maximum prime number to test. Default is 20000.

    Returns:
    - dict: A dictionary containing the results of the benchmark.
    """
    try:
        # Run the sysbench CPU test
        command = [
            "sysbench",
            "--test=cpu",
            f"--cpu-max-prime={max_prime}",
            f"--num-threads={num_threads}",
            "run"
        ]
        
        # Execute the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        # Parse the output for relevant information
        output = result.stdout
        return {
            "total_time": parse_output(output, "total time:"),
            "min_time_per_request": parse_output(output, "min:"),
            "avg_time_per_request": parse_output(output, "avg:"),
            "max_time_per_request": parse_output(output, "max:"),
            "events_per_second": parse_output(output, "events per second:")
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def parse_output(output, keyword):
    """
    Helper function to extract a specific value from sysbench output.

    Parameters:
    - output (str): The sysbench command output.
    - keyword (str): The keyword to search for in the output.

    Returns:
    - str: The extracted value associated with the keyword.
    """
    lines = output.splitlines()
    for line in lines:
        if keyword in line:
            return line.split(":")[-1].strip()
    return None

# Example usage:
results = test_cpu_performance(num_threads=4, max_prime=20000)
if results:
    print("CPU Benchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
