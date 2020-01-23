import torch
import numpy as np
import random
import timeit

class ProxyNS(torch.nn.Module):

    def __init__(self, sz_embed, classeslist, sigma, subsampling = None):
        super().__init__()
        self.sz_embed = sz_embed
        self.classeslist = classeslist
        self.classesdict = {}
        for idx, cnum in enumerate(classeslist):
            self.classesdict[cnum] = idx
        self.sigma = sigma
        self.subsampling = subsampling
        self.proxies = torch.nn.Embedding(len(classeslist), sz_embed)
        torch.nn.init.xavier_uniform_(self.proxies.weight)

        self.dist = lambda x, y: torch.pow(
            torch.nn.PairwiseDistance(eps=1e-16)(x, y),
            2
        )


    def nca(self, xs, ys, i):
      
        # NOTE possibly something wrong with labels/classes ...
        # especially, if trained on labels in range 100 to 200 ... 
        # then y = ys[i] can be for example 105, but Z has 0 to 100
        # therefore, all proxies become negativ!

        x = xs[i] # embedding of sample i, produced by embedded bninception
        y = ys[i] # label of sample i
        # for Z: of all labels, select those unequal to label y
        # do subsampling
        #tt1 = timeit.default_timer()
        if self.subsampling is None:
            Z = torch.tensor(self.classeslist).long()
        else:
            Zcadi = random.sample(self.classeslist, int(len(self.classeslist) * self.subsampling))
            if y not in Zcadi:
                Zcadi.append(y)
            Z = torch.tensor(Zcadi).long()
        #tt2 = timeit.default_timer()
        #print('subsampling : {} sec'.format(tt2-tt1))


        # all classes/proxies
        #assert Z.size(0) == int(len(self.classeslist) * self.subsampling)

        # with proxies embedding, select proxy i for target, p(ys)[i] <=> p(y)

        #tt1 = timeit.default_timer()

        p_dist_temp = self.dist(
                torch.nn.functional.normalize(
                    self.proxies(torch.tensor(self.classesdict[y.item()]).long().cuda()),
                    # [1, 64], normalize along dim = 1 (64)
                    dim=0
                ),
                x.unsqueeze(0)
            )

        #tt2 = timeit.default_timer()
        #print('p_dist_temp : {} sec'.format(tt2-tt1))


        #tt1 = timeit.default_timer()
        #p_dist shape = (1, 1)
        p_dist = torch.exp(
            - torch.div(p_dist_temp, self.sigma)      # temperature scaling
        )
        #tt2 = timeit.default_timer()
        #print('p_dist : {} sec'.format(tt2-tt1))


        #tt1 = timeit.default_timer()

        Z = torch.tensor([self.classesdict[zz] for zz in Z.cpu().numpy()])
        #tt2 = timeit.default_timer()
        #print('Z to tensor : {} sec'.format(tt2-tt1))

        #tt1 = timeit.default_timer()
        n_dist_temp = self.dist(
                torch.nn.functional.normalize(
                    self.proxies(Z.cuda().long()),  # [nb_classes - 1, 64]
                    dim=1
                ),
                x.expand(Z.size(0), x.size(0))  # [nb_classes - 1, 64]
            )   # [embed, nb_classes]

        #tt2 = timeit.default_timer()
        #print('n_dist_temp : {} sec'.format(tt2-tt1))

        #n_dist shape = (1, nb_classes)
        #tt1 = timeit.default_timer()

        n_dist = torch.exp(
            - torch.div(n_dist_temp, self.sigma)  # temperature scaling
        )

        #tt2 = timeit.default_timer()
        #print('n_dist : {} sec'.format(tt2-tt1))

        return -torch.log(p_dist / torch.sum(n_dist))


      
    def forward(self, xs, ys):
        return torch.mean(
            torch.stack(
                [self.nca(xs, ys, i) for i in range(len(ys))]
            )
        )
