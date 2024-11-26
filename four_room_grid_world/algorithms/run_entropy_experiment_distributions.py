import subprocess

NUMBER_RUNS = 10

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--goal-distribution", "standard_uniform", "--tag", "PPO_RLE_STANDARD_UNIFORM"]
    subprocess.run(["python", "ppo_rle_adopted.py"] + arguments)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--goal-distribution", "von_mises", "--tag", "PPO_RLE_VON_MISES"]
    subprocess.run(["python", "ppo_rle_adopted.py"] + arguments)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}", "--goal-distribution", "exponential", "--tag", "PPO_RLE_EXPONENTIAL"]
    subprocess.run(["python", "ppo_rle_adopted.py"] + arguments)
    print(f"Ended run {i}")