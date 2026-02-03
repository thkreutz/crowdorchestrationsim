### based on https://shchur.github.io/blog/2021/tpp2-neural-tpps/

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = "cuda"
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = "cpu"

class Weibull:
    """Weibull distribution.
    
    Args:
        b: scale parameter b (strictly positive)
        k: shape parameter k (strictly positive)
        eps: Minimum value of x, used for numerical stability.
    """
    def __init__(self, b, k, eps=1e-8):
        self.b = b
        self.k = k
        self.eps = eps
    
    def log_prob(self, x):
        """Logarithm of the probability density function log(f(x))."""
        # x must have the same shape as self.b and self.k
        x = x.clamp_min(self.eps)  # pow is unstable for inputs close to 0
        return (self.b.log() + self.k.log() + (self.k - 1) * x.log() 
                + self.b.neg() * torch.pow(x, self.k))
    
    def log_survival(self, x):
        """Logarithm of the survival function log(S(x))."""
        x = x.clamp_min(self.eps)
        return self.b.neg() * torch.pow(x, self.k)
    
    def sample(self, sample_shape=torch.Size()):
        """Generate a sample from the distribution."""
        # We do sampling using the inverse transform method
        # If z ~ Expo(1), then solving exp(-z) = S(x) for x produces 
        # a sample from the distribution with survival function S
        shape = torch.Size(sample_shape) + self.b.shape
        z = torch.empty(shape).exponential_(1.0)
        return (z * self.b.reciprocal() + self.eps).pow(self.k.reciprocal())

class NeuralTPP(nn.Module):
    """A simple neural TPP model with an RNN encoder.
    
    Args:
        context_size: Size of the RNN hidden state.
    """
    def __init__(self, context_size=32):
        super().__init__()
        self.context_size = context_size
        # Used to embed the event history into a context vector
        self.rnn = nn.GRU(
            input_size=2, 
            hidden_size=context_size, 
            batch_first=True,
        )
        # Used to obtain model parameters from the context vector
        self.hypernet = nn.Linear(
            in_features=context_size, 
            out_features=2,
        )
    
    def get_context(self, inter_times):
        """Get context embedding for each event in each sequence.
        
        Args:
            inter_times: Padded inter-event times, shape (B, L)
            
        Returns:
            context: Context vectors, shape (B, L, C)
        """
        tau = inter_times.unsqueeze(-1)
        # Clamp tau to avoid computing log(0) for padding and getting NaNs
        log_tau = inter_times.clamp_min(1e-8).log().unsqueeze(-1)  # (B, L, 1)
        
        rnn_input = torch.cat([tau, log_tau], dim=-1)
        # The intial state is automatically set to zeros
        rnn_output = self.rnn(rnn_input)[0]  # (B, L, C)
        # Shift by one such that context[:, i] will be used
        # to parametrize the distribution of inter_times[:, i]
        context = F.pad(rnn_output[:, :-1, :], (0, 0, 1, 0))  # (B, L, C)
        return context
    
    def get_inter_time_distribution(self, context):
        """Get context embedding for each event in each sequence.
        
        Args:
            context: Context vectors, shape (B, L, C)
            
        Returns:
            dist: Conditional distribution over the inter-event times
        """
        raw_params = self.hypernet(context)  # (B, L, 2)
        b = F.softplus(raw_params[..., 0])  # (B, L)
        k = F.softplus(raw_params[..., 1])  # (B, L)
        return Weibull(b=b, k=k)
    
    def nll_loss(self, inter_times, seq_lengths):
        """Compute negative log-likelihood for a batch of sequences.
        
        Args:
            inter_times: Padded inter_event times, shape (B, L)
            seq_lengths: Number of events in each sequence, shape (B,)
        
        Returns:
            log_p: Log-likelihood for each sequence, shape (B,)
        """
        context = self.get_context(inter_times)  # (B, L, C)
        inter_time_dist = self.get_inter_time_distribution(context)

        log_pdf = inter_time_dist.log_prob(inter_times)  # (B, L)
        # Construct a boolean mask that selects observed events
        arange = torch.arange(inter_times.shape[1], device=seq_lengths.device)
        mask = (arange[None, :] < seq_lengths[:, None]).float()  # (B, L)
        log_like = (log_pdf * mask).sum(-1)  # (B,)

        log_surv = inter_time_dist.log_survival(inter_times)  # (B, L)
        end_idx = seq_lengths.unsqueeze(-1)  # (B, 1)
        log_surv_last = torch.gather(log_surv, dim=-1, index=end_idx)  # (B, 1)
        log_like += log_surv_last.squeeze(-1)  # (B,)
        return -log_like
    
    def sample(self, batch_size, t_end, device="cuda"):
        """Generate an event sequence from the TPP.
        
        Args:
            batch_size: Number of samples to generate in parallel.
            t_end: Time until which the TPP is simulated.
        
        Returns:
            inter_times: Padded inter-event times, shape (B, L)
            seq_lengths: Number of events in each sequence, shape (B,)
        """
        inter_times = torch.empty([batch_size, 0]).to(device)
        next_context = torch.zeros(batch_size, 1, self.context_size).to(device)
        generated = False

        while not generated:

            next_context = next_context.clamp_min(1e-8)
            inter_time_dist = self.get_inter_time_distribution(next_context)
            next_inter_times = inter_time_dist.sample()  # (B, 1)

            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (B, L)

            # Obtain the next context vector
            tau = next_inter_times.unsqueeze(-1)  # (B, 1, 1)
            log_tau = next_inter_times.clamp_min(1e-8).log().unsqueeze(-1)  # (B, 1, 1)
            rnn_input = torch.cat([tau, log_tau], dim=-1)  # (B, 1, 2)
            next_context = self.rnn(rnn_input, next_context.transpose(0, 1))[0]  # (B, 1, C)

            # Check if the end of the interval has been reached
            generated = inter_times.sum(-1).min() >= t_end

            

        # Convert the sample to the same format as our input data
        arrival_times = inter_times.cumsum(-1)
        seq_lengths = (arrival_times < t_end).sum(-1).long() 
        inter_times = arrival_times - F.pad(arrival_times, (1, 0))[..., :-1]
        return inter_times, seq_lengths