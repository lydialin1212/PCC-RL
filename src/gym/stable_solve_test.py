# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default


input_size = 5

class CustomNetwork(torch.nn.Module):
    def __init__(self, input_size=input_size):
        super().__init__()
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )
        self.value_net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.Linear(32, 16),
            torch.nn.Linear(16, 1),
            torch.nn.Tanh()
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)



global TrainCustomNetwork

class MyMlpPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 custom_param, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args,
                         **kwargs)
        print('custom_param=', custom_param)
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        TrainCustomNetwork = CustomNetwork()
        self.mlp_extractor = TrainCustomNetwork


env = gym.make('PccNs-v0')


gamma = arg_or_default("--gamma", default=0.99)
print("gamma = %f" % gamma)
model = PPO(MyMlpPolicy, env, verbose=1, batch_size=2048, gamma=gamma)


MODEL_PATH = "./pcc_model_%d.pt"
for i in range(0, 6):
    model.learn(total_timesteps=(1600 * 410))
    torch.save(TrainCustomNetwork.state_dict(), MODEL_PATH % i)



