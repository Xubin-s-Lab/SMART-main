from scipy import stats
import torch
from tqdm import tqdm

from layer import SMART,SAGEConv_Decoder,SAGEConv_Encoder
import torch.nn.functional as F
import numpy as np

def laplacian_regularization(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    loss = (diff ** 2).sum(dim=1).mean()
    return loss

def train_SMART(features,
                edges,
                triplet_samples_list,
                weights=[1, 1],
                emb_dim=64, 
                n_epochs=500,
                lr=0.0001,
                weight_decay=1e-5,
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                window_size=20,
                slope=0.0001,
                Conv_Encoder=SAGEConv_Encoder,
                Conv_Decoder=SAGEConv_Decoder,
                margin=0.5,
                return_loss=False,
                laplacian_alpha=0
                ): 
    
    hidden_dims=[x.shape[1] for x in features]+[emb_dim]
    
    model = SMART(hidden_dims=hidden_dims,device=device,Conv_Encoder=Conv_Encoder,Conv_Decoder=Conv_Decoder)
  
    features, edges= [x.to(device) for x in features], [edge.to(device) for edge in edges]
    model.to(device)
    
    n_epochs = n_epochs
    loss_list = []
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)

    best_rec_loss = float('inf')
    best_tri_loss = float('inf')
    rec_loss_no_improve = 0
    tri_loss_no_improve = 0

    window_size = window_size  # 检测窗口长度

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, x_rec= model(features, edges)

        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')

        tri_loss=0
        for i,(anchors,positives,negatives) in enumerate(triplet_samples_list):
            anchor_arr = z[anchors,]
            positive_arr = z[positives,]
            negative_arr = z[negatives,]
            
            tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
            w=weights[len(weights)//2+i]
            tri_loss+=w*tri_output

        rec_loss=0
        for i,(feature,x_r) in enumerate(zip(features,x_rec)):
            rec_output=F.mse_loss(feature, x_r)
            w=weights[i]
            rec_loss+=weights[i]*rec_output

        
        loss = rec_loss  + tri_loss

        if laplacian_alpha!=0:
            loss+=laplacian_alpha*laplacian_regularization(z, edges[0])

        if epoch > window_size and epoch%10==0:
            # 计算最近窗口的线性回归斜率p值
            x = np.arange(window_size)
            res1 = stats.linregress(x, [i[1] for i in loss_list[-window_size:]])
            res2 = stats.linregress(x, [i[2] for i in loss_list[-window_size:]])
            if abs(res1.slope) <slope or abs(res2.slope) <slope:  # 无显著下降趋势
                if res1.slope!=0 and res2.slope!=0:
                    print(f"Stopping for flat trend")
                    break

                
        loss_list.append((loss.item(),tri_loss.item(),rec_loss.item()))
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
   
    return model if return_loss==False else (model,loss_list)
