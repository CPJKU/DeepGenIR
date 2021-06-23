"""A generic NCE wrapper which speedup the training and inferencing"""
"""code originated from https://github.com/Stonesjtu/Pytorch-NCE/tree/master/nce"""

import math
from math import isclose

import torch
import torch.nn as nn
import torch.nn.functional as F


# A backoff probability to stabilize log operation
BACKOFF_PROB = 1e-10


class OutputLayerProbs(nn.Module):
    """
    There are 3 modes in this module:
        - nce: enable the Noise Contrastive Estimation approximation. NCE is to eliminate the computational cost of softmax normalization.
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default

    Ref:
        X.Chen etal Recurrent neural network language
        model training with noise contrastive estimation
        for speech recognition
        https://core.ac.uk/download/pdf/42338485.pdf

    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        mode: calculation mode -> 'full', 'sampled', 'nce'

    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`

    Input:
        target: the supervised training label.
        input: input to be passed to the child class (usually the linear layer)

    Return:
        logprob: a scalar log probability by default, :math:`(B, N)` if `reduction='none'`
    """

    def __init__(self,
                 noise,
                 noise_ratio=100,
                 norm_term='auto',
                 mode_type='nce',
                 ):
        super(OutputLayerProbs, self).__init__()

        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        probs = noise / noise.sum()
        probs = probs.clamp(min=BACKOFF_PROB)
        renormed_probs = probs / probs.sum()

        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

        self.noise_ratio = noise_ratio
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term

        #self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mode_type = mode_type

    def forward(self, input, target):

        batch_size = target.size(0)
        max_len = target.size(1)
        
        if self.mode_type == 'full':
            # predicted is excpected to be log probabilities
            logprobs = self.nll_logprobs(input, target)
        else:

            noise_samples = self.get_noise(batch_size, max_len)

            # B,N,Nr
            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            # (B,N), (B,N,Nr)
            logit_target_in_model, logit_noise_in_model = self._get_logit(input, target, noise_samples)
            
            if self.mode_type in ['nce', 'ncelog', 'ncesigmoid', 'ncelogsig']:
                if self.training:
                    logprobs = self.nce_logprobs(logit_target_in_model, logit_noise_in_model, 
                                                 logit_noise_in_noise, logit_target_in_noise,
                    )
                else:
                    # output the approximated posterior
                    logprobs = logit_target_in_model
                    logprobs = torch.sum(logprobs, dim=1)
            elif self.mode_type == 'sampled':
                logprobs = self.sampled_softmax_logprobs(logit_target_in_model, logit_noise_in_model,
                                                         logit_noise_in_noise, logit_target_in_noise,
                )
            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError(
                    'mode type {} not implemented at {}'.format(
                        self.mode_type, current_stage
                    )
                )


        return logprobs

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        noise_size = (batch_size, max_len, self.noise_ratio)
        noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, input, target_idx, noise_idx):
        """Get the logits of NCE estimated probability for target and noise

        Both NCE and sampled_softmax are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.

        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_logit, noise_logit = self.get_score(input, target_idx, noise_idx)

        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        return target_logit, noise_logit

    def get_score(self, input, target_idx, noise_idx):
        """Get the target and noise score

        Usually logits are used as score.
        This method should be override by inherit classes

        Returns:
            - target_score: real valued score for each target index
            - noise_score: real valued score for each noise index
        """
        raise NotImplementedError()

    def nce_logprobs(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the log probabilities given all four probabilities

        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution

        Returns:
            - logprobs: log probabilities for every single case
        """

        # NOTE: prob <= 1 is not guaranteed
        logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        # predicted probability of the word comes from true data distribution
        # The posterior can be computed as following
        # p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
        # For numeric stability we compute the logits of true label and
        # directly use bce_with_logits.
        # Ref https://pytorch.org/docs/stable/nn.html?highlight=bce#torch.nn.BCEWithLogitsLoss
        #logit_true = logit_model - logit_noise - math.log(self.noise_ratio)
        p_true = logit_model.exp() / (logit_model.exp() + self.noise_ratio * logit_noise.exp())
        
        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1

        #logprobs = -self.bce_with_logits(logit_true, label)
        logprobs = -nn.BCELoss(reduction='none')(p_true, label)
        logprobs = torch.sum(logprobs.view(logprobs.size(0), -1), dim=1)
        
        return logprobs

    def sampled_softmax_logprobs(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax based on the tensorflow's impl"""
        logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)
        # subtract Q for correction of biased sampling
        logits = logits - q_logits
        logproba = torch.nn.LogSoftmax(dim=-1)(logits)
        
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
        
        logprobs = -torch.nn.NLLLoss(reduction='none')(logproba.view(-1, logproba.size(-1)), labels.view(-1)).view_as(labels)
        logprobs = torch.sum(logprobs, dim=1)
        
        return logprobs
    
    def nll_logprobs(self, input, target_idx):
        """Get the conventional NLL
        The returned logprobs should be of the same size of `target`
        Args:
            - target_idx: batched target index
            - args, kwargs: any arbitrary input if needed by sub-class
        Returns:
            - logprobs: the estimated logprob for each target
        """
        raise NotImplementedError()
    
    
class OutputLinear(OutputLayerProbs):
    """A linear layer that only decodes the results of provided indices
    Args:
        target_idx: indices of target words
        noise_idx: indices of noise words
        input: input matrix
    Shape:
        - target_idx :math:`(B, N)` where `max(M) <= N` B is batch size
        - noise_idx :math:`(B, N, N_r)` where `max(M) <= N`
        - Input :math:`(B, N, in\_features)`
    Return:
        - target_score :math:`(N, 1)`
        - noise_score :math:`(N, N_r)` the un-normalized score
    """

    def __init__(self, hidden_dim, num_classes, *args, **kwargs):
        super(OutputLinear, self).__init__(*args, **kwargs)
        
        # use Embedding to store the output embedding
        # it's efficient when it comes sparse update of gradients
        self.emb = nn.Embedding(num_classes, hidden_dim)
        self.bias = nn.Embedding(num_classes, 1)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.emb.embedding_dim)
        self.emb.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            # initialize the bias with unigram instead of uniform
            self.bias.weight.data = torch.unsqueeze(
                self.logprob_noise + self.norm_term, 1
            )

    def get_score(self, input, target_idx, noise_idx):
        """
        Shape:
            - target_idx: :math:`(B, L)` where `B` is batch size
            `L` is sequence length
            - noise_idx: :math:`(B, L, N_r)` where `N_r is noise ratio`
            - input: :math:`(B, L, E)` where `E = output embedding size`
        """

        return self._compute_sampled_logit_batched(input, target_idx, noise_idx)


    def _compute_sampled_logit_batched(self, input, target_idx, noise_idx):
        """compute the logits of given indices based on input vector
        A batched version, it speeds up computation and puts less burden on
        sampling methods.
        Args:
            - target_idx: :math:`B, L, 1` flatten to `(N)` where `N=BXL`
            - noise_idx: :math:`B, L, N_r`, noises at the dim along B and L
            should be the same, flatten to `N_r`
            - input: :math:`(B, L, E)` where `E = vector dimension`
        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)

        target_batch = self.emb(target_idx)
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_bias = self.bias(target_idx).squeeze(1)  # N
        target_score = torch.sum(input * target_batch, dim=1) + target_bias  # N X E * N X E

        noise_batch = self.emb(noise_idx)  # Nr X H
        # noise_bias = self.bias.index_select(0, noise_idx).unsqueeze(0)  # Nr
        noise_bias = self.bias(noise_idx)  # 1, Nr
        noise_score = torch.matmul(
            input, noise_batch.t()
        ) + noise_bias.t()  # N X Nr
        return target_score.view(original_size), noise_score.view(*original_size, -1)

    def nll_logprobs(self, input, target_idx):
        score = F.linear(input, self.emb.weight, self.bias.weight.squeeze(1))  # (B, L, V)
        logproba = torch.nn.LogSoftmax(dim=-1)(score)
        
        logproba = logproba.transpose(1, 2)
        logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(logproba, target_idx)
        logprobs = torch.sum(logprobs, dim=1)

        return logprobs
    
    
class AliasMultinomial(torch.nn.Module):
    '''Alias sampling method to speedup multinomial sampling
    The alias method treats multinomial sampling as a combination of uniform sampling and
    bernoulli sampling. It achieves significant acceleration when repeatedly sampling from
    the save multinomial distribution.
    Attributes:
        - probs: the probability density of desired multinomial distribution
    Refs:
        - https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    def __init__(self, probs):
        super(AliasMultinomial, self).__init__()

        assert isclose(probs.sum().item(), 1), 'The noise distribution must sum to 1'
        cpu_probs = probs.cpu()
        K = len(probs)

        # such a name helps to avoid the namespace check for nn.Module
        self_prob = [0] * K
        self_alias = [0] * K

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for idx, prob in enumerate(cpu_probs):
            self_prob[idx] = K*prob
            if self_prob[idx] < 1.0:
                smaller.append(idx)
            else:
                larger.append(idx)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self_alias[small] = large
            self_prob[large] = (self_prob[large] - 1.0) + self_prob[small]

            if self_prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self_prob[last_one] = 1

        self.register_buffer('prob', torch.Tensor(self_prob))
        self.register_buffer('alias', torch.LongTensor(self_alias))

    def draw(self, *size):
        """Draw N samples from multinomial
        Args:
            - size: the output size of samples
        """
        max_value = self.alias.size(0)

        kk = self.alias.new(*size).random_(0, max_value).long().view(-1)
        prob = self.prob[kk]
        alias = self.alias[kk]
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob).long()
        oq = kk.mul(b)
        oj = alias.mul(1 - b)

        return (oq + oj).view(size)
    