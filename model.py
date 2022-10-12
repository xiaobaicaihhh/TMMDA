class TMMDA(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(TMMDA, self).__init__()
        self.fc1 = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.fc2 = nn.Linear(VISUAL_DIM, TEXT_DIM)
        
        # self.enc = nn.Sequential(nn.Linear(TEXT_DIM, TEXT_DIM), nn.Linear(TEXT_DIM, TEXT_DIM))
        # self.gen = nn.Sequential(nn.Linear(TEXT_DIM, TEXT_DIM), nn.Linear(TEXT_DIM, TEXT_DIM))
        # self.discriminator = nn.Sequential(MAGPooler(dim=TEXT_DIM), nn.Linear(TEXT_DIM, 2))
        self.mixup = TokenMixUp()
    
        self.generator = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        ) 

        self.discriminator = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(TEXT_DIM // 2, 2),
        )

        
        self.adv_loss = nn.CrossEntropyLoss()
        self.js_loss = JSD()
        
        encoder_a = nn.TransformerEncoderLayer(d_model=TEXT_DIM, nhead=8)
        self.trans_a = nn.TransformerEncoder(encoder_a, num_layers=3)
        encoder_v = nn.TransformerEncoderLayer(d_model=TEXT_DIM, nhead=8)
        self.trans_v = nn.TransformerEncoder(encoder_v, num_layers=3)
        

        self.attn = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.1)
        
        # self.pooler_l = MAGPooler(dim=TEXT_DIM)
        # self.pooler_a = MAGPooler(dim=TEXT_DIM)
        # self.pooler_v = MAGPooler(dim=TEXT_DIM)
        self.fc_lm = nn.Linear(2*TEXT_DIM, TEXT_DIM)
    def forward(self, text_embedding, visual, acoustic, input_mask_mix=None, training=True):
        bsz, seq, dim = text_embedding.shape
        #torch.Size([48, 50, 768]) torch.Size([48, 50, 47]) torch.Size([48, 50, 74])
        xl, xv, xa = text_embedding, visual, acoustic
        xv, xa = self.fc2(xv), self.fc1(xa)
        # add gan
        mask = torch.rand(bsz, seq).to(DEVICE)
        xlm = self.mixup(xl, xv, xa, mask, input_mask_mix)
        #training noise gen
        if training:
            xm_noise = alpha_noise*torch.randn(bsz, seq, dim).to(DEVICE) #+ xm_noise
            xlmg = self.generator(xlm) + xm_noise
        else:
            xlmg = self.generator(xlm)
        # discrimilator
        xdl = self.discriminator(xl)
        xdm = self.discriminator(xlmg)
        # label
        text_label_real = torch.ones(bsz*50, dtype=torch.long).to(DEVICE)
        text_label_fake = torch.zeros(bsz*50, dtype=torch.long).to(DEVICE)
        # loss
        adv_loss = self.adv_loss(xdl.view(-1, 2), text_label_real) + self.adv_loss(xdm.view(-1, 2), text_label_fake)
        js_loss = self.js_loss(xl, xlm)

        # transformer encoder
        xl, xlmg = xl.transpose(0, 1), xlmg.transpose(0, 1)
        xl = self.attn(xl, xlmg, xlmg)[0] + xl
        #xl = self.fc_lm(torch.cat([xclmg, xl], dim=-1))
        xl = xl.transpose(0, 1)
        return xl, adv_loss, js_loss

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.p_softmax = nn.Softmax(dim=-1)
        self.q_softmax = nn.Softmax(dim=-1)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p = self.p_softmax(p)
        q = self.q_softmax(q)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
    
class TokenMixUp(nn.Module):
    def __init__(self):
        super(TokenMixUp, self).__init__()
    def forward(self, x0, x1, x2, mask, input_masks, prob1=0.85, prob2=0.15):
        input_mask = input_masks >= 1
        mask0 = (mask > prob2) & (mask < prob1) & input_mask
        mask1 = mask > prob1
        mask2 = mask < prob2
        mask1 = mask1 & input_mask
        mask2 = mask2 & input_mask
        x = torch.zeros_like(x0)
        x[mask0] = x0[mask0]
        x[mask1] = x1[mask1]
        x[mask2] = x2[mask2]
        return x