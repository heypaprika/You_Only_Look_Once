
import torch

class D_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dist = lambda x, y: torch.pow(
            torch.nn.PairwiseDistance(eps=1e-16)(x, y),
            2
        )

    def distance(self, anchor_v, pos_v, neg_v):

        # anchor_v - pos_v
        p_distance = self.dist(anchor_v, pos_v)

        # anchor_v - neg_v
        n_distance = self.dist(anchor_v, neg_v)

        return torch.sum(
            p_distance, torch.div(1, n_distance)
        )

    def forward(self, anchors, poss, negs):
        return torch.mean(
            torch.stack(
                [self.distance(anchors[i], poss[i], negs[i]) for i in range(len(anchors))]
            )
        )