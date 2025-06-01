import torch
import torch.nn as nn
import torch.nn.functional as F


class C_Attention(nn.Module):
    """Constructs a Channel, Height, and Weight Attention module.
    Input: Batch size x Channel x Height x Weight: 1x96x20x20
    Args:
        kernel size - k: Adaptive selection of kernel size
        output: 1x96x1x1
    """
    def __init__(self, c2, k=3): # mac du khong dung c2 nhung ghi cho thong nhat ECA, G_A, W_A
        super(C_Attention, self).__init__()
        # For Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False) # [batch_size, channels, length]
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1x96x20x20
        y = self.avg_pool(x)    # y: 1x96x1x1

        # squeeze(-1): remove last dimension: 1x96x1
        # transpose(-1, -2): swap 2 last dimensions: 1x1x96
        # conv1d(1,1,3,1): kernel size=3 > size(h,w) cua 1x96x1, tuc h=w=1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.silu(y)
        y = self.sigmoid(y)

        # return x * y.expand_as(x) # use for MyHWC, MyHWC_v1, MyHWC_v2
        # If y is of shape (1, C, 1, 1) and x is of shape (N, C, H, W), 
        # then y.expand_as(x) will expand y to (N, C, H, W) by replicating its values along the new dimensions.
        return y    # B x C x 1 x 1
    
class H_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1 x 2048 x 20 x 1
    """
    def __init__(self, c1, c2, k=3):
        super(H_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Output size: [H, 1]
        self.conv = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1 x 1024 x 20 x 20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, H, 1] 1 x 1024 x 20 x 1
        y = y.squeeze(-1)     # y: [B, C, H]: 1 x 1024 x 20
        
        # # Prepare for Conv1d by permuting dimensions
        # y = y.permute(0, 2, 1)  # y: [B, H, C]
        
        # # Multi-scale information fusion
        y = self.conv(y)  # [B, C, H'] 1 x 2048 x 20
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, H, 1]
        # y = y.permute(0, 2, 1)  # [B, C, H', 1]
        y = y.unsqueeze(-1)     # [B, C, H', 1] 1 x 2048 x 20 x 1
        
        return y  # Output shape: [B, C, H', 1]: 1 x 2048 x 20 x 1

class W_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1 x 2048 x 1 x 20
    """
    def __init__(self, c1, c2, k=3):
        super(W_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Output size: [1, W]
        self.conv = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1 x 1024 x 20 x 20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, 1, W] 1 x 1024 x 1 x 20
        y = y.squeeze(-2)     # y: [B, C, W]: 1 x 1024 x 20
        
        # # Multi-scale information fusion
        y = self.conv(y)  # [B, C, W'] 1 x 2048 x 20
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, 1, W]
        y = y.unsqueeze(-2)     # [B, C, 1, W'] 1 x 2048 x 1 x 20
        
        return y  # Output shape: [B, C, 1, W']: 1 x 2048 x 1 x 20

class CBS(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, activation)
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
           
class Spatial_Attention(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.mp5 = nn.MaxPool2d(kernel_size=5, stride=2, padding=5//2)
        self.mp9 = nn.MaxPool2d(kernel_size=9, stride=2, padding=9//2)
        self.mp13 = nn.MaxPool2d(kernel_size=13, stride=2, padding=13//2)
        self.conv = nn.Conv2d(c1*3, c2, 1, 1)   # after concate 3 maxpoling
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU()
        
    def forward(self, x):
        # print("# Input Size: ", x.size())                   # B x C1 x H x W
        h_att = self.ha(x)                                  
        # print("# Height Output Size: ", h_att.size())       # B x C1 x H x 1
        w_att = self.wa(x)                                  
        # print("# Weight Output Size: ", w_att.size())       # B x C1 x 1 x W
        hw_att = x * h_att * w_att                             
        # print("# H*W Output Size: ", hw_att.size())         # B x C1 x H x W
        
        mpl5 = self.mp5(x)
        # print("# MP5 Output Size: ", mpl5.size())           # B x C1 x H/2 x W/2
        mpl9 = self.mp9(x)
        # print("# MP9 Output Size: ", mpl9.size())           # B x C1 x H/2 x W/2
        mpl13 = self.mp13(x)
        # print("# MP13 Output Size: ", mpl13.size())         # B x C1 x H/2 x W/2
        
        cat = torch.cat((mpl5,mpl9,mpl13),dim=1)
        # print("# CAT Output Size: ", cat.size())            # B x 3*C1 x H/2 x W/2
        con = self.conv(cat)
        # print("# CONV Output Size: ", con.size())           # B x C2 x H/2 x W/2
        spacedown = self.sl(self.bn(con))
        # print("# SPD Output Size: ", spacedown.size())      # B x C2 x H/2 x W/2
        return spacedown

class ScaleDotProduct(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.ca = C_Attention(c2)
        
        
    def forward(self, x):
        # print("# Input Size: ", x.size())                   # B x C1 x H x W
        h_att = self.ha(x)                                  
        # print("# Height Output Size: ", h_att.size())       # B x C1 x H x 1
        w_att = self.wa(x)                                  
        # print("# Weight Output Size: ", w_att.size())       # B x C1 x 1 x W
        c_att = self.ca(x)                                  
        # print("# Channel Output Size: ", c_att.size())      # B x C1 x 1 x 1
        
        # matmul (H,W)
        BQ,CQ,HQ,WQ = h_att.size()
        Q = h_att.view(BQ,CQ,HQ*WQ)
        # print("# Q Output Size: ", Q.size())                # B x C1 x H*W
        
        # Interpolate w_att to have the same height and width as h_att
        w_att = nn.functional.interpolate(w_att, size=(h_att.size(2), h_att.size(3)), mode='bilinear', align_corners=False)
        
        
        # BK,CK,HK,WK = w_att.size()
        # K = w_att.view(BK,CK,HK*WK)
        K = w_att.view(BQ, CQ, HQ * WQ)
        # print("# K Output Size: ", K.size())                # B x C1 x H*W
        
        # Compute the attention scores by performing matrix multiplication of Q and K
        K_transpose = K.transpose(1,2)
        # print("# K_transpose Output Size: ", K_transpose.size())    # B x H*W x C1
        
        scores = torch.bmm(Q, K_transpose)
        # print("# scores Output Size: ", scores.size())              # B x C1 x C1
        
        # Scale the scores
        d_k = Q.size(-1)  # This is the depth (H*W)
        scores_scaled = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # Scale by sqrt(d_k)
        # print("# Scaled scores Output Size: ", scores_scaled.size())     # B x C1 x C1
        
        # Apply softmax to get the attention weights
        attention_weights = F.softmax(scores_scaled, dim=-1)  # Shape: [Batch, Channels, Channels]
        # print("# Attention weights Output Size: ", attention_weights.size())     # B x C1 x C1

        # Multiply the attention weights with the value vector V
        BV,CV,HV,WV = c_att.size()
        V = c_att.view(BV,CV,HV*WV)
        
        output = torch.bmm(attention_weights, V)    # Shape: [Batch, Channels, 1]
        # print("# Output Size: ", output.size())     # B x C1 x C1

        # Reshape output to match the original spatial dimensions
        output = output.view(BV, CV, 1, 1)
        # print("# Output Reshape Size: ", output.size())     # B x C1 x 1 x 1
        
        # trans from into input shape
        output = x * output
        # print("# Final Reshape Size: ", output.size())     # B x C1 x H x W
        
        return output  

class Contigous_Att(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.sdp = ScaleDotProduct(c1, c2)
        self.conv = nn.Conv2d(c1*5, c2, 1, 1)   # after concate 3 maxpoling
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU()
        
    def forward(self, x):
        # print("# Input Size: ", x.size())                   # B x C1 x H x W
        y1 = self.sdp(x)                                  
        # print("# y1 Output Size: ", y1.size())              # B x C1 x H x W
        y2 = self.sdp(y1)                                  
        # print("# y2 Output Size: ", y2.size())              # B x C1 x H x W
        y3 = self.sdp(y2)                                  
        # print("# y3 Output Size: ", y3.size())              # B x C1 x H x W
        y4 = self.sdp(y1)                                  
        # print("# y4 Output Size: ", y4.size())              # B x C1 x H x W
        output = torch.cat((y1,y2,y3,y4,x),dim=1)
        # print("# Output Size: ", output.size())             # B x C1*5 x H x W
        output = self.sl(self.bn(self.conv(output)))
       # print("# Output Final Size: ", output.size())        # B x C1*5 x H x W
        return output   