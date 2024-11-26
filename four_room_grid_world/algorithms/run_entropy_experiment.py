import subprocess

NUMBER_RUNS = 10

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}"]
    subprocess.run(["python", "ppo.py"] + arguments)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}"]
    subprocess.run(["python", "ppo_noisy_net.py"] + arguments)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}"]
    subprocess.run(["python", "ppo_rle_adopted.py"] + arguments)
    print(f"Ended run {i}")

for i in range(NUMBER_RUNS):
    print(f"Starting run {i}")
    seed = i
    arguments = ["--track", "--seed", f"{seed}"]
    subprocess.run(["python", "ppo_rnd.py"] + arguments)
    print(f"Ended run {i}")
