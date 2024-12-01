def record_episode(env, agent, device, max_steps=200, filename="episode.gif"):
    frames = []
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to(device)

    for step in range(max_steps):
        frames.append(env.render())

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
            action = action.cpu().item()

        obs, _, terminated, truncated, _ = env.step(action)
        obs = torch.Tensor(obs).to(device)

        if terminated or truncated:
            break

    imageio.mimsave(filename, frames, duration=1000 / 30)


def record_episode_with_probs(env, agent, device, max_steps=200, filename="episode.gif", iteration=0):
    frames = []
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to(device)

    # Store action probabilities for plotting
    all_probs = []
    accumulated_reward = 0

    for step in range(max_steps):
        frames.append(env.render())

        with torch.no_grad():
            # Get action and probabilities
            logits = agent.actor(obs)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            action = torch.distributions.Categorical(logits=logits).sample()

            # Store probabilities
            all_probs.append(probs.cpu().numpy())

            action = action.cpu().item()

        obs, reward, terminated, truncated, _ = env.step(action)
        accumulated_reward += reward
        obs = torch.Tensor(obs).to(device)

        if terminated or truncated:
            break

    # Save the GIF
    imageio.mimsave(filename, frames, duration=1000 / 30)

    # Plot probability histogram
    if len(all_probs) > 0:
        all_probs = np.array(all_probs)
        # Calculate the average probability for each action across all steps
        action_counts = all_probs.sum(axis=0)  # Sum probabilities for each action

        plt.figure(figsize=(10, 5))
        num_actions = len(action_counts)
        plt.bar(range(num_actions), action_counts / len(all_probs))
        plt.title(f'Action Distribution - Iteration {iteration}\nAccumulated Reward: {accumulated_reward:.2f}')
        plt.xlabel('Action')
        plt.ylabel('Average Probability')
        plt.xticks(range(num_actions))
        plt.grid(True)

        # Save the plot
        plot_filename = filename.replace('.gif', '_probs.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f"Episode {iteration} - Accumulated Reward: {accumulated_reward:.2f}")