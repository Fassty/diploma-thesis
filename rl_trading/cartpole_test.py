import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

ray.init(local_mode=True)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=0)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    .build()
)

for i in range(50):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")