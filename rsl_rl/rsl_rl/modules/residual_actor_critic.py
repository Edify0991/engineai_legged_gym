# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import get_activation


class CombinedActor(nn.Module):
    def __init__(
        self,
        base_actor,
        residual_actor,
        residual_scale=1.0,
        residual_clip=1.0,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.residual_actor = residual_actor
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)

    def forward(self, observations):
        base_actions = self.base_actor(observations)
        residual_actions = self.residual_actor(observations)
        residual_actions = torch.clamp(residual_actions, -self.residual_clip, self.residual_clip)
        return base_actions + self.residual_scale * residual_actions

    @torch.jit.export
    def infer_base_and_residual(self, observations):
        base_actions = self.base_actor(observations)
        residual_actions = self.residual_actor(observations)
        clipped_residual_actions = torch.clamp(
            residual_actions, -self.residual_clip, self.residual_clip
        )
        combined_actions = base_actions + self.residual_scale * clipped_residual_actions
        return base_actions, residual_actions, clipped_residual_actions, combined_actions


class ResidualActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs_dim,
        critic_obs_dim,
        action_dim,
        base_ckpt_path,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        residual_hidden_dims=[256, 256, 256],
        activation='elu',
        init_noise_std=1.0,
        freeze_base=True,
        residual_scale=1.0,
        residual_clip=1.0,
        residual_last_layer_gain=0.01,
        map_location='cpu',
        **kwargs,
    ):
        if kwargs:
            print(
                'ResidualActorCritic.__init__ got unexpected arguments, which will be ignored: '
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        base_actor = self._build_mlp(obs_dim, action_dim, actor_hidden_dims, activation)
        self._load_base_actor_from_ckpt(base_actor, base_ckpt_path, map_location)

        if freeze_base:
            for param in base_actor.parameters():
                param.requires_grad = False

        residual_actor = self._build_mlp(
            obs_dim,
            action_dim,
            residual_hidden_dims,
            activation,
        )
        self._init_residual_last_layer(
            residual_actor,
            gain=residual_last_layer_gain,
        )

        self.base_actor = base_actor
        self.residual_actor = residual_actor
        self.actor = CombinedActor(
            base_actor=self.base_actor,
            residual_actor=self.residual_actor,
            residual_scale=residual_scale,
            residual_clip=residual_clip,
        )

        self.critic = self._build_mlp(
            critic_obs_dim,
            1,
            critic_hidden_dims,
            activation,
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def _build_mlp(input_dim, output_dim, hidden_dims, activation_name):
        layers = [nn.Linear(input_dim, hidden_dims[0]), get_activation(activation_name)]
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(get_activation(activation_name))
        return nn.Sequential(*layers)

    @staticmethod
    def _load_base_actor_from_ckpt(base_actor, base_ckpt_path, map_location):
        checkpoint = torch.load(base_ckpt_path, map_location=map_location)
        if 'model_state_dict' not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dict': " + str(base_ckpt_path)
            )

        model_state_dict = checkpoint['model_state_dict']
        actor_state_dict = {
            key[len('actor.') :]: value
            for key, value in model_state_dict.items()
            if key.startswith('actor.')
        }

        if len(actor_state_dict) == 0:
            raise KeyError("No actor.* weights found in checkpoint model_state_dict")

        missing_keys, unexpected_keys = base_actor.load_state_dict(actor_state_dict, strict=False)
        if missing_keys:
            raise RuntimeError(
                'Missing base actor keys while loading checkpoint: ' + str(missing_keys)
            )
        if unexpected_keys:
            raise RuntimeError(
                'Unexpected base actor keys while loading checkpoint: ' + str(unexpected_keys)
            )

    @staticmethod
    def _init_residual_last_layer(residual_actor, gain):
        last_linear = None
        for module in residual_actor.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise RuntimeError('Residual actor has no Linear layer to initialize')

        nn.init.xavier_uniform_(last_linear.weight, gain=gain)
        nn.init.constant_(last_linear.bias, 0.0)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    @torch.no_grad()
    def infer_base_and_residual(self, observations):
        base_actions = self.base_actor(observations)
        residual_actions = self.residual_actor(observations)
        clipped_residual_actions = torch.clamp(
            residual_actions, -self.actor.residual_clip, self.actor.residual_clip
        )
        combined_actions = base_actions + self.actor.residual_scale * clipped_residual_actions
        return base_actions, residual_actions, clipped_residual_actions, combined_actions

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
