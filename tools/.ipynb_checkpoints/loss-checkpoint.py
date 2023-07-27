import torch
import torch.nn as nn
from sentence_transformers import util
import torch.nn.functional as F

class NTXent(nn.Module):

    def __init__(self, temperature=0.07):
        super(NTXent, self).__init__()
        self.loss = nn.LogSoftmax(dim=1)
        self.tau = temperature

    def forward(self, audio_embeds, text_embeds, labels):

        n = audio_embeds.shape[0]

        a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(a2t.device)
        mask_diag = mask.diag()
        mask_diag = torch.diag_embed(mask_diag)
        mask = mask ^ mask_diag

        a2t_loss = - self.loss(a2t).masked_fill(mask, 0).diag().mean()
        t2a_loss = - self.loss(t2a).masked_fill(mask, 0).diag().mean()
        
        '''
        #a2t = util.cos_sim(audio_embeds, text_embeds) / self.tau
        #t2a = util.cos_sim(text_embeds, audio_embeds) / self.tau
        #t2a_loss = -self.loss(t2a).mean()
        #a2t_loss = -self.loss(a2t).mean()
        '''

        loss = 0.5 * a2t_loss + 0.5 * t2a_loss

        return loss




class TripletLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds:
        :param text_embeds:
        :param labels:
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        # clear diagonals
        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)
        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        cost_s = cost_s.max(1)[0]
        cost_a = cost_a.max(0)[0]

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss


class BiDirectionalRankingLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(BiDirectionalRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, audio_embeds, text_embeds, labels):
        """

        :param audio_embeds: (batch_size, embed_dim)
        :param text_embeds: (batch_size, embed_dim)
        :param labels: (batch_size, )
        :return:
        """

        n = audio_embeds.size(0)  # batch size

        # dist = []
        sim_a2t = util.cos_sim(audio_embeds, text_embeds)  # (batch_size, x batch_size)
        sim_ap = torch.diag(sim_a2t).view(n, 1)
        d1 = sim_ap.expand_as(sim_a2t)
        d2 = sim_ap.t().expand_as(sim_a2t)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = F.relu(self.margin + sim_a2t - d1)
        # compare every diagonal score to scores in its row
        # audio retrieval
        cost_a = F.relu(self.margin + sim_a2t - d2)

        mask = labels.expand(n, n).eq(labels.expand(n, n).t()).to(cost_a.device)

        cost_s = cost_s.masked_fill(mask, 0)
        cost_a = cost_a.masked_fill(mask, 0)

        loss = (cost_s.sum() + cost_a.sum()) / n

        return loss




class WeightTriplet(nn.Module): #코드 이상함 이거
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2):
        super(WeightTriplet, self).__init__()
        self.margin = margin

    def polyloss(self, sim_mat, label):
        epsilon = 1e-5
        size = sim_mat.size(0)
        hh = sim_mat.t()

        loss = list()
        for i in range(size):
            pos_pair_ = sim_mat[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)

            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / size
        return loss

    def forward(self, audio_embeds, text_embeds, labels):
        # compute image-sentence score matrix
        scores = util.cos_sim(audio_embeds, text_embeds)
        loss = self.polyloss(scores, labels)
        return loss
    

class VICReg(nn.Module):

    def __init__(
        self,
        inv_weight=1.0,
        var_weight=1.0,
        cov_weight=0.04
    ):
        super().__init__()

        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, audio_embs, caption_embs, labels=None):
        
        Z_a, Z_b = audio_embs, caption_embs

        N, D = Z_a.size()

        # Invariance loss
        inv_loss = F.mse_loss(Z_a, Z_b)

        # Variance loss
        Z_a_std = torch.sqrt(Z_a.var(dim=0) + 1e-04)
        Z_b_std = torch.sqrt(Z_b.var(dim=0) + 1e-04)
        var_loss = torch.mean(F.relu(1 - Z_a_std))
        var_loss += torch.mean(F.relu(1 - Z_b_std))

        # Covariance loss
        Z_a = Z_a - Z_a.mean(dim=0)
        Z_b = Z_b - Z_b.mean(dim=0)
        Z_a_cov = (Z_a.T @ Z_a) / (N - 1)
        Z_b_cov = (Z_b.T @ Z_b) / (N - 1)
        
        diag = torch.eye(D, dtype=torch.bool, device=Z_a.device)
        cov_loss = Z_a_cov[~diag].pow_(2).sum() / D
        cov_loss += Z_b_cov[~diag].pow_(2).sum() / D

        loss = self.inv_weight * inv_loss
        loss += self.var_weight * var_loss
        loss += self.cov_weight * cov_loss
        return loss
    

class InfoNCE(nn.Module):
    
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        
        self.tau = temperature
    
    def forward(self, audio_embeds, text_embeds, labels=None):
        """
        :param audio_embeds: tensor, (N,E)
        :param text_embeds
        :param labels(item_batch) # audio-text info
        :return:
        """
        n = audio_embeds.size(0) # 배치 사이즈

        similarity_matrix = util.cos_sim(audio_embeds,text_embeds)

        similarity_prob_matrix = F.log_softmax(similarity_matrix.exp()/self.tau,dim=-1)

        loss = - similarity_prob_matrix.diag().sum()

        return loss / n



class InfoNCE_VICReg(nn.Module):
    
    def __init__(self, info_weight=1, vic_weight=1):
        super(InfoNCE_VICReg, self).__init__()
        
        self.info_weight= info_weight
        self.vic_weight = vic_weight
        self.InfoNCE = InfoNCE()
        self.VICReg = VICReg()

    def forward(self, audio_embeds, text_embeds, labels=None):

        loss1 = self.InfoNCE(audio_embeds, text_embeds)
        loss2 = self.VICReg(audio_embeds, text_embeds)

        return self.info_weight * loss1 + self.vic_weight * loss2