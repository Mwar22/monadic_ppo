# -*- coding:utf-8 -*-
###
# File:  core.py
# Created Date: 29/05/2026 01:06:35
# Author: Lucas de Jesus  (lucasdejesusphysic@gmail.com)
# -----
# Last Modified: 03/06/2026 05:40:14
# Modified By: Lucas de Jesus 
# -----
# Copyright (c) 2026
# 
# This file is subject to the terms and conditions defined in
# the 'LICENSE.txt' file found in the root of this source tree.
# Please read LICENSE.txt for full copyright and licensing details.
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import jax
from mujoco import mjx
from typing import Protocol, Tuple, Dict, Any, runtime_checkable
from flax import nnx, struct
from .enviroment import MujocoEnv
from abc import ABC, abstractmethod

@runtime_checkable
class Policy(Protocol):
    def __call__(self, obs: jax.Array)->jax.Array | Tuple[jax.Array, ...]:
        """
        Forward pass para a politica

        Parameters
        ----------
        obs : jax.Array

        Returns
        -------
        jax.Array | Tuple[jax.Array, ...]
        """
        ...
    
    @property
    def obs_size(self)->int:
        ...
        
    @property
    def action_size(self)->int:
        ...
        
    def sample(self, obs: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """
        Realiza a coleta de uma ação dado uma observação.

        Parameters
        ----------
        obs : jax.Array
        rngs : nnx.Rngs

        Returns
        -------
        jax.Array
            ação amostrada
        """
        ...

    def evaluate_actions(self, obs: jax.Array, actions: jax.Array)->Tuple[jax.Array, jax.Array]:
        """
        Avalia uma dada ação tomada a partir de uma dada observação, segundo os parametros atuais.
        
        Parameters
        ----------
        obs : jax.Array
        actions : jax.Array

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            logprobs e entropia associada
        """
        ...

@runtime_checkable
class Value(Protocol):
    def __call__(self, obs: jax.Array)->jax.Array:
        """_summary_

        Parameters
        ----------
        obs : jax.Array
            _description_

        Returns
        -------
        jax.Array
            _description_
        """
        ...
    @property
    def obs_size(self)->int:
        ...


class ResetData(struct.PyTreeNode):
    action: jax.Array
    logprob: jax.Array
    value: jax.Array
    entropy: jax.Array

class StepData(struct.PyTreeNode):
    reward: jax.Array
    done: jax.Array
    info: Dict[str, Any]
    
#@runtime_checkable
class Agent(nnx.Module, ABC):

    @property
    @abstractmethod
    def policy(self)->Policy:
        ...

    @property
    @abstractmethod
    def value(self)-> Value:
        ...

    @staticmethod
    @abstractmethod
    def compose_obs(env: MujocoEnv, mjx_data:mjx.Data)->Tuple[jax.Array, jax.Array]:
        """
        Deve coletar informações de um ambiente 'MujocoEnv' e compor observações para a política e para a função de valor.
        
        Parameters
        ----------
        env : MujocoEnv
        mjx_data : mjx.Data

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Tupla de jax.Array. O primeiro representa as observações para a política. O segundo para as observações da função de valor.
        """
        ...

    @abstractmethod
    def reset(self, env: MujocoEnv,  rngs: nnx.Rngs, mjx_data:mjx.Data)->Tuple[ResetData, mjx.Data]:
        ...

    @abstractmethod
    def step(self, env: MujocoEnv,  action: jax.Array, target: jax.Array, mjx_data:mjx.Data,)->Tuple[StepData, mjx.Data]:
        ...