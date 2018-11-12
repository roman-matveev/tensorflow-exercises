import gym

env = gym.make('CartPole-v0')

print('Observation:')
obs = env.reset()
print(obs)

if __name__ == '__main__':
    for _ in range(2):
        # env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("\n\tPerformed One Random Action")
        print('\nobs:')
        print(obs)

        print('\nreward:')
        print(reward)

        print('\ndone:')
        print(done)

        print('\ninfo:')
        print(info)
