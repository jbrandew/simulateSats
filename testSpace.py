import pstats

# Load profiling results
profile_stats = pstats.Stats('profile_results.prof')

# Calculate total time
total_time = profile_stats.total_tt

# Filter functions where cumulative time is greater than 1% of total time
profile_stats.strip_dirs().sort_stats('cumulative').print_stats(.01)

# Alternative method using a specific function
profile_stats.print_stats(10)
