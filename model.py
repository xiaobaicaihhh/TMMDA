class TMMDA(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):
        super(TMMDA, self).__init__()
        self.fc1 = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.fc2 = nn.Linear(VISUAL_DIM, TEXT_DIM)
        
        # self.enc = nn.Sequential(nn.Linear(TEXT_DIM, TEXT_DIM), nn.Linear(TEXT_DIM, TEXT_DIM))
        # self.gen = nn.Sequential(nn.Linear(TEXT_DIM, TEXT_DIM), nn.Linear(TEXT_DIM, TEXT_DIM))
        # self.discriminator = nn.Sequential(MAGPooler(dim=TEXT_DIM), nn.Linear(TEXT_DIM, 2))
        self.mixup = TokenMixUp()
    
        self.generator_l = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        ) 
        self.discriminator_l = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(TEXT_DIM // 2, 2),
        )

        self.generator_v = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        ) 
        self.discriminator_v = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(TEXT_DIM // 2, 2),
        )

        self.generator_a = nn.Sequential(
            nn.Linear(TEXT_DIM, TEXT_DIM),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(TEXT_DIM, TEXT_DIM)
        ) 
        self.discriminator_a = nn.Sequential(
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
        

        self.attn1 = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.1)
        self.attn2 = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.1)
        self.attn3 = MultiheadAttention(embed_dim=TEXT_DIM, num_heads=8, attn_dropout=0.1)
        
        self.trans = TransformerEncoder(embed_dim=TEXT_DIM, num_heads=8, layers=1)
        
    def forward(self, text_embedding, visual, acoustic, input_mask_mix=None, training=True):
        bsz, seq, dim = text_embedding.shape
        #torch.Size([48, 50, 768]) torch.Size([48, 50, 47]) torch.Size([48, 50, 74])
        xl, xv, xa = text_embedding, visual, acoustic
        xv, xa = self.fc2(xv), self.fc1(xa)
        

        
        # add gan
        mask = torch.rand(bsz, seq).to(DEVICE)
        xlm = self.mixup(xl, xv, xa, mask, input_mask_mix, prob1=rho_1, prob2=rho_2)
        xvm = self.mixup(xv, xl, xa, mask, input_mask_mix, prob1=rho_1, prob2=rho_2)
        xam = self.mixup(xa, xv, xa, mask, input_mask_mix, prob1=rho_1, prob2=rho_2)
        if training:
            xlm_noise = alpha_noise*torch.randn(bsz, seq, dim).to(DEVICE) #+ xm_noise
            xlmg = self.generator_l(xlm) + xlm_noise

            xvm_noise = alpha_noise*torch.randn(bsz, seq, dim).to(DEVICE) #+ xm_noise
            xvmg = self.generator_v(xvm) + xvm_noise

            xam_noise = alpha_noise*torch.randn(bsz, seq, dim).to(DEVICE) #+ xm_noise
            xamg = self.generator_a(xam) + xam_noise            
        else:
            xlmg = self.generator_l(xlm)
            xvmg = self.generator_v(xvm)
            xamg = self.generator_a(xam)
            
        xdl = self.discriminator_l(xl)
        xdlmg = self.discriminator_l(xlmg)
        
        xdv = self.discriminator_v(xv)
        xdvmg = self.discriminator_v(xvmg)
        
        xda = self.discriminator_a(xa)
        xdamg = self.discriminator_a(xamg)
                
        text_label_real = torch.ones(bsz*50, dtype=torch.long).to(DEVICE)
        text_label_fake = torch.zeros(bsz*50, dtype=torch.long).to(DEVICE)
        
        adv_loss_l = self.adv_loss(xdl.view(-1, 2), text_label_real) + self.adv_loss(xdlmg.view(-1, 2), text_label_fake)
        adv_loss_v = self.adv_loss(xdv.view(-1, 2), text_label_real) + self.adv_loss(xdvmg.view(-1, 2), text_label_fake)
        adv_loss_a = self.adv_loss(xda.view(-1, 2), text_label_real) + self.adv_loss(xdamg.view(-1, 2), text_label_fake)
        adv_loss = (adv_loss_l +  adv_loss_v + adv_loss_a) / 3
        
        js_loss_l = self.js_loss(xl, xlmg)
        js_loss_v = self.js_loss(xv, xvmg)
        js_loss_a = self.js_loss(xa, xamg)
        js_loss = (js_loss_l + js_loss_v + js_loss_a) / 3
        # transformer encoder
        #ret_xl, ret_xlmg, ret_xv, ret_xvmg, ret_xa, ret_xamg = xl[:, 0, :], xlmg[:, 0, :], xv[:,0,:], xvmg[:,0,:], xa[:,0,:], xamg[:,0,:]
        ret_xl, ret_xlmg, ret_xv, ret_xvmg, ret_xa, ret_xamg = torch.sum(xl, dim=-1), torch.sum(xlmg, dim=-1), torch.sum(xv, dim=-1), torch.sum(xvmg, dim=-1), torch.sum(xa, dim=-1), torch.sum(xamg, dim=-1)

        xl, xlmg = xl.transpose(0, 1), xlmg.transpose(0, 1)
        xl = self.attn1(xl, xlmg, xlmg)[0] + xl + text_embedding.transpose(0, 1)
        xl = xl.transpose(0, 1)
        
        xv, xvmg = xv.transpose(0, 1), xvmg.transpose(0, 1)
        xv = self.attn2(xv, xvmg, xvmg)[0] + xv
        xv = xv.transpose(0, 1)
        
        xa, xamg = xa.transpose(0, 1), xamg.transpose(0, 1)
        xa = self.attn3(xa, xamg, xamg)[0] + xa
        xa = xa.transpose(0, 1)

        x = torch.cat([xl, xv, xa], dim=1)
        x = x.transpose(0, 1)
        x = self.trans(x)
        x = x.transpose(0, 1)
        return x
    
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
