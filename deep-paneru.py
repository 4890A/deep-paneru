import retro
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gym

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "custom_integrations")
)
print("PanelDePon" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
base_env = retro.make("PanelDePon", inttype=retro.data.Integrations.ALL)

env = Monitor(base_env, 'monitor.csv')
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
#model.load('paneru_agent.zip')

env = gym.wrappers.Monitor(base_env, directory='./results', force=True)
obs = env.reset()
for i in range(1000000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
model.save('paneru_agent2')
