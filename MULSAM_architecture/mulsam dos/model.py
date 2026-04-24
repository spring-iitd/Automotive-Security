import torch
import torch.nn as nn

class EnhancedSelfAttention(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=24):
        super(EnhancedSelfAttention, self).__init__()
        self.query = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.key   = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_map = torch.sigmoid(Q * K) 
        weighted_V = attn_map * V
        out = self.output_conv(weighted_V)
        return out.squeeze(1)

class MULSAM(nn.Module):
    def __init__(self, input_size=11, hidden_size=24, num_classes=2):
        super(MULSAM, self).__init__()
        self.attention = EnhancedSelfAttention(input_size, hidden_size)
        self.lstm_time = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_depth = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        x_enhanced = self.attention(x)
        out_t, (h_t, _) = self.lstm_time(x_enhanced)
        out_d, (h_d, _) = self.lstm_depth(x_enhanced)
        combined = torch.cat((h_t[-1], h_d[-1]), dim=1)
        return self.fc(combined)