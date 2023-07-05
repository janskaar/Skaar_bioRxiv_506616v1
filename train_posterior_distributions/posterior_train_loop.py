import torch
import numpy as np
import os

MAX_EPOCHS = 5000
STOP_PERSISTENCE = 100      # number of epochs to train without loss improvement before stopping
STOP_THRESHOLD = 0.01       # value to consider "not significantly improving"

def loss_fn(flow, eval_likelihood, y0, num=1024):
    samples = flow.sample(torch.Size((num,))).permute(0,1)
    # dirty hack with permute, for some reason calling flow.log_prob
    # on samples directly is run without gradients, applying random op
    # first will disable this. why???

    log_q = flow.log_prob(samples)

    # numerical instabilities (?) close to edge of prior some times give
    # very high (~1e20) log_probs, check and discard those
    keep_inds = torch.abs(log_q) < 20.
    samples = samples[keep_inds]
    log_q = log_q[keep_inds]
    with torch.no_grad():
        log_p = eval_likelihood(samples, y0)
        logweights = log_p - log_q
        logweights -= logweights.max()
        weights = logweights.exp()
        weights /= weights.sum()
    loss = -torch.sum(weights * (log_p - log_q))
    loss_q = -torch.sum(weights * log_q)
    return loss_q, loss

def train(model, eval_likelihood, y0, opt, trainable_params, num_particles=256, logfile=None, savefile=None):
    if logfile:
        with open(logfile, "w") as f:
            pass
    max_loss = 1e10
    loss_qs = []
    losses = []
    for i in range(MAX_EPOCHS):
        opt.zero_grad()
        loss_q, loss = loss_fn(model, eval_likelihood, y0, num=num_particles)

        # Numerical issues (?) with flow some times causes extreme negative values
        # for log q, if this happens just go to next epoch to avoid nans
#         if loss_q.item() > 10000:
#             print("SKIPPING")
#             continue
        loss_q.backward()
#         torch.nn.utils.clip_grad_norm_(trainable_params, 10.)
        opt.step()
        current_loss = loss_q.item()
        loss_qs.append(current_loss)
        losses.append(loss.item())
        
        if logfile:
            with open(logfile, "a") as f:
                f.write(f"{i},{loss_qs[-1]},{losses[-1]}\n")
        # If improvement, save model 
        if current_loss < (max_loss - STOP_THRESHOLD):
            if savefile:
                torch.save(model, savefile)
            max_loss = current_loss
            rounds_wo_improvements = 0

        # if no improvement for STOP_PERSISTENCE epochs, stop training
        else:
            rounds_wo_improvements += 1
            if rounds_wo_improvements > STOP_PERSISTENCE:
                print("NO IMPROVEMENT, BREAKING")
                break

    return loss_qs, losses
