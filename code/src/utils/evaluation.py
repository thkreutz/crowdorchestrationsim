import torch

def compute_ade(prediction, target):
    # compute l2 distance for each positions, sum, and divide by number of steps
    ADE = torch.sum((prediction-target).pow(2).sum(1).sqrt()) / len(prediction)
    return ADE

### FDE
def compute_fde(prediction, target):
    # l2 distance for last position
    FDE = (prediction-target).pow(2).sum(1).sqrt()[-1]
    return FDE

def compute_ade_fde_seq(agent_dict, transform, log=False, in_pixels=False):
    seq_ade = 0
    seq_fde = 0
    n_seqs = 0
    #in_pixels = False
    for k in agent_dict.keys():

        if in_pixels:
            targ = transform.get_pixel_positions_torch(agent_dict[k]["target"]).detach().cpu()
            pred = transform.get_pixel_positions_torch(agent_dict[k]["preds"]).detach().cpu()
        else:
            targ = torch.tensor(agent_dict[k]["target"]).detach().cpu()
            pred = torch.tensor(agent_dict[k]["preds"]).detach().cpu()

        ade = compute_ade(pred, targ)
        fde = compute_fde(pred, targ)

        seq_ade += ade
        seq_fde += fde
        n_seqs += 1
        
    if log:
        print("ADE=%s, \r\n FDE=%s" % (seq_ade/n_seqs, seq_fde/n_seqs))
    
    #print(seq_ade, seq_fde, n_seqs)
    if seq_ade == 0 or seq_fde == 0 or n_seqs == 0:
        return 0, 0
    else:
        return seq_ade / n_seqs, seq_fde / n_seqs