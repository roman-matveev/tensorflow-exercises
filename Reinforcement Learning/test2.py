import gym

env = gym.make('CartPole-v0')
obs = env.reset()

if __name__ == '__main__':
    for _ in range(1000):
        env.render()
        cart_pos, cart_vel, pole_angle, angle_vel = obs

        if pole_angle > 0:
            action = 1
        else:
            action = 0

        obs, reward, done, info = env.step(action)
