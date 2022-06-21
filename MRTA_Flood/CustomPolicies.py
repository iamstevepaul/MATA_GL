"""
Author: Steve Paul 
Date: 1/18/22 """
import torch

from stable_baselines3.common.policies import BasePolicy
import torch as th
import gym
import math
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)
from typing import NamedTuple
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

#   TODO:
#   Make the policy network task independent

def dicttuple(cls: tuple):
    """Extends a tuple class with methods for the dict constructor."""

    cls.keys = lambda self: self._fields
    cls.__getitem__ = _getitem
    return cls

def _getitem(instance, index_or_key):
    """Returns the respective item."""

    if isinstance(index_or_key, str):
        try:
            return getattr(instance, index_or_key)
        except AttributeError:
            raise IndexError(index_or_key) from None

    return super().__getitem__(index_or_key)

@dicttuple
class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: th.Tensor
    context_node_projected: th.Tensor
    glimpse_key: th.Tensor
    glimpse_val: th.Tensor
    logit_key: th.Tensor

    # def __new__(cls, name, bases, namespace):
    #     my_fancy_new_namespace = {'__module__': module}
    #     if '__classcell__' in namespace:
    #         my_fancy_new_namespace['__classcell__'] = namespace['__classcell__']
    #     return super().__new__(cls, name, bases, my_fancy_new_namespace)


class ActorCriticGCAPSPolicy(BasePolicy):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[th.nn.Module] = th.nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 device: Union[th.device, str] = "auto"
                 ):
        super(ActorCriticGCAPSPolicy, self).__init__(observation_space,
                                                     action_space,
                                                     features_extractor_class,
                                                     features_extractor_kwargs,
                                                     optimizer_class=optimizer_class,
                                                     optimizer_kwargs=optimizer_kwargs,
                                                     squash_output=squash_output)

        features_dim = features_extractor_kwargs['features_dim']
        node_dim = features_extractor_kwargs['node_dim']
        agent_node_dim = features_extractor_kwargs['agent_node_dim']
        self.node_dim = features_extractor_kwargs['node_dim']

        value_net_net = [th.nn.Linear(features_dim, features_dim, bias=False),
                         th.nn.Linear(features_dim, 1, bias=False)]
        self.value_net = th.nn.Sequential(*value_net_net).to(device=device)
        if features_extractor_kwargs['feature_extractor'] == "CAPAM":
            from Feature_Extractors import CAPAM_P
            self.features_extractor = CAPAM_P(
                node_dim=node_dim,
                features_dim=features_dim,
                K=features_extractor_kwargs['K'],
                Le=features_extractor_kwargs['Le'],
                P=features_extractor_kwargs['P'],
                device=device
            ).to(device=device)
        elif features_extractor_kwargs['feature_extractor'] == "MLP":
            from Feature_Extractors import MLP
            inter_dim = features_dim * (features_extractor_kwargs['K'] + 1) * features_extractor_kwargs['P']
            self.features_extractor = MLP(
                node_dim=node_dim,
                features_dim=features_dim,
                inter_dim=inter_dim,
                device=device
            ).to(device=device)
        elif features_extractor_kwargs['feature_extractor'] == "AM":
            from Feature_Extractors import GraphAttentionEncoder
            self.features_extractor = GraphAttentionEncoder(
                node_dim=node_dim,
                n_heads=features_extractor_kwargs['n_heads'],
                embed_dim=features_dim,
                n_layers=features_extractor_kwargs['Le'],
                device=device
            ).to(device=device)
        self.agent_decision_context = th.nn.Linear(agent_node_dim, features_dim).to(device=device)
        self.agent_context = th.nn.Linear(agent_node_dim, features_dim).to(device=device)
        self.full_context_nn = th.nn.Linear(2 * features_dim, features_dim).to(device=device)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

        self.project_fixed_context = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.project_node_embeddings = th.nn.Linear(features_dim, 3 * features_dim, bias=False).to(device=device)
        self.project_out = th.nn.Linear(features_dim, features_dim, bias=False).to(device=device)
        self.n_heads = features_extractor_kwargs['n_heads']
        self.tanh_clipping = features_extractor_kwargs['tanh_clipping']
        self.mask_logits = features_extractor_kwargs['mask_logits']
        self.temp = features_extractor_kwargs['temp']


    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, deterministic=deterministic)
        return th.tensor([actions])

    def _build(self):
        pass

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        # values = self.value_net(latent_vf)
        distribution, values = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()


    def forward(self, obs, deterministic=False,  *args, **kwargs):
        torch.cuda.reset_max_memory_allocated(device=None)

        # latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        # values = self.value_net(latent_vf)

        distribution, values = self.get_distribution(obs)

        deterministic = True
        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = th.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        # if self.mask_inner:
        #     assert self.mask_logits, "Cannot mask inner without masking logits"
        #     compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = th.matmul(th.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = (th.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))) ### steve has made a change here (normalizing)

        ## From the logits compute the probabilities by clipping, masking and softmax
        # if self.tanh_clipping > 0:
        #     logits = th.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[th.tensor(mask[:,:,:].reshape(logits.shape), dtype=th.bool)] = -math.inf
        # if mask[0, 0,0] == 1:
        #     logits[:,:,0] = -math.inf
            # print("depot masked")
        # print("Shape of logits: ", logits.shape)
        return logits, glimpse.squeeze(-2)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )


    def _get_attention_node_data(self, fixed):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def decode_action_probabilites(self, latent_pi, graph_embed, features, obs, num_steps=1):
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(features[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        fixed = AttentionModelFixed(features, fixed_context, *fixed_attention_node_data)
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)

        query = fixed.context_node_projected + latent_pi
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, obs['mask'])

        log_p = th.log_softmax(log_p / self.temp, dim=-1)

        return log_p


    def get_distribution(self, obs):


        features, graph_embed = self.extract_features(obs)

        latent_pi, values = self.context_extractor(graph_embed, obs)
        mean_actions = self.decode_action_probabilites(latent_pi,graph_embed, features, obs)[:,0,:]
        # mean_actions = self.action_net(latent_pi)
        latent_sde = latent_pi
        # if self.sde_features_extractor is not None:
        #     latent_sde = self.sde_features_extractor(features)


        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        # values = self.value_net(latent_vf)
        return distribution, values

    def context_extractor(self, graph_embed, observations):
        agent_taking_decision = observations['agent_taking_decision'][:,0].to(torch.int64)
        n_data = agent_taking_decision.shape[0]
        context = graph_embed[:, None, :] \
               + \
                  self.full_context_nn(
                      th.cat((self.agent_decision_context(observations['agents_graph_nodes'][torch.arange(0,n_data), agent_taking_decision, :][:,None,:]),
                              self.agent_context(observations['agents_graph_nodes']).sum(1)[:,None,:]), -1))
        return context, self.value_net(context)
        # return context, self.value_net(th.cat((self.agent_decision_context(observations['agent_taking_decision_coordinates']),
        #                       self.agent_context(observations['agents_destination_coordinates']).sum(1)[:,None,:], observations['mask'].reshape((mask_shape[0],mask_shape[2], mask_shape[1]))), -1))
    # def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_sde: Optional[th.Tensor] = None) -> Distribution:
    #     mean_actions = self.action_net(latent_pi)
    #
    #     if isinstance(self.action_dist, DiagGaussianDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std)
    #     elif isinstance(self.action_dist, CategoricalDistribution):
    #         # Here mean_actions are the logits before the softmax
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, MultiCategoricalDistribution):
    #         # Here mean_actions are the flattened logits
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, BernoulliDistribution):
    #         # Here mean_actions are the logits (before rounding to get the binary actions)
    #         return self.action_dist.proba_distribution(action_logits=mean_actions)
    #     elif isinstance(self.action_dist, StateDependentNoiseDistribution):
    #         return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
    #     else:
    #         raise ValueError("Invalid action distribution")