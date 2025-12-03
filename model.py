import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Custom CNN Layer (From Scratch using Im2Col)
# ==========================================
class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CustomConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Manually define weights and bias
        # Shape: (Out, In, K, K)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming Initialization (Crucial for convergence)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x shape: (Batch, In_Channels, H, W)
        batch_size, in_c, h, w = x.shape
        
        # Calculate Output Dimensions
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # 1. Im2Col (Unfold): Extract patches
        # Output: (Batch, In_Channels * K * K, L)
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        # 2. Flatten Weights
        # Shape: (Out_Channels, In_Channels * K * K)
        weight_flat = self.weight.view(self.out_channels, -1)
        
        # 3. Matrix Multiplication (Convolution)
        # (Out, In*K*K) @ (Batch, In*K*K, L) -> (Batch, Out, L)
        # We perform matmul per batch item or using broadcast
        # x_unfolded needs transpose for simple matmul: (Batch, L, In*K*K) is easier? 
        # Actually, standard broadcast: weight @ x_unfolded works if we align dims properly.
        # Efficient way: output = weight_flat @ x_unfolded
        # Since weight_flat is (Out, Dim) and x_unfolded is (Batch, Dim, L)
        # We can treat weight as (1, Out, Dim) and matmul
        
        # Using PyTorch matmul broadcasting:
        output = weight_flat @ x_unfolded 
        
        # 4. Add Bias
        output += self.bias.view(1, -1, 1)
        
        # 5. Fold (Reshape)
        output = output.view(batch_size, self.out_channels, h_out, w_out)
        
        return output

# ==========================================
# 2. Custom LSTM Layer (From Scratch using Gates)
# ==========================================
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weights for all 4 gates (Input, Forget, Cell, Output)
        # x -> gates
        self.W_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        # h -> gates
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, state):
        h_prev, c_prev = state
        
        # Gates calculation: Linear transforms
        gates = F.linear(x, self.W_ih.t(), self.bias_ih) + \
                F.linear(h_prev, self.W_hh.t(), self.bias_hh)
        
        # Split into i, f, g, o
        i, f, g, o = gates.chunk(4, 1)
        
        # Apply Activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Update Cell State (Long-term memory)
        c_cur = (f * c_prev) + (i * g)
        
        # Update Hidden State (Short-term memory)
        h_cur = o * torch.tanh(c_cur)
        
        return h_cur, c_cur

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # We are implementing a single layer LSTM for simplicity
        self.cell = CustomLSTMCell(input_size, hidden_size)

    def forward(self, x, init_states=None):
        # x shape: (Batch, Seq_Len, Input_Size)
        batch_size, seq_len, _ = x.shape
        
        if init_states is None:
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h, c = init_states
            
        outputs = []
        
        # Manual Loop over Sequence Length
        for t in range(seq_len):
            x_step = x[:, t, :]
            h, c = self.cell(x_step, (h, c))
            outputs.append(h.unsqueeze(1))
            
        return torch.cat(outputs, dim=1), (h, c)

# ==========================================
# 3. Integrated Model (Architecture Wrapper)
# ==========================================
class CustomImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embedding_weights=None):
        super(CustomImageCaptionModel, self).__init__()
        
        # --- A. Custom CNN Encoder ---
        # Replacing nn.Conv2d with CustomConv2d
        
        # Layer 1
        self.conv1 = CustomConv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Standard BN is usually allowed
        
        # Layer 2
        self.conv2 = CustomConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3
        self.conv3 = CustomConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer 4
        self.conv4 = CustomConv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Flatten: 256 channels * 14 * 14 spatial
        self.fc_img = nn.Linear(256 * 14 * 14, embed_size)
        
        # --- B. Emotion Embedding ---
        # 9 emotions mapped to vector
        self.emotion_embed = nn.Embedding(9, embed_size)
        
        # --- C. Custom LSTM Decoder ---
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if embedding_weights is not None:
            self.embed.weight.data.copy_(embedding_weights)
            self.embed.weight.requires_grad = True
            
        # Input Size Calculation:
        # Image (Embed) + Emotion (Embed) + Word (Embed)
        self.lstm = CustomLSTM(input_size=embed_size * 3, hidden_size=hidden_size)
        
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward_encoder(self, images):
        # Pass through custom Conv blocks
        x = self.pool(self.relu(self.bn1(self.conv1(images))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        features = self.dropout(self.relu(self.fc_img(x)))
        return features

    def forward(self, images, captions, emotions):
        # 1. Image Features (Batch, Embed)
        img_features = self.forward_encoder(images)
        
        # 2. Emotion Features (Batch, Embed)
        emo_features = self.emotion_embed(emotions)
        
        # 3. Word Embeddings (Batch, Seq, Embed)
        embeddings = self.dropout(self.embed(captions))
        
        # 4. Concatenate Static Context (Image + Emotion)
        # (Batch, Embed*2)
        context = torch.cat((img_features, emo_features), dim=1)
        
        # 5. Expand Context to match Sequence Length
        # (Batch, 1, Embed*2) -> (Batch, Seq, Embed*2)
        context_expanded = context.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        
        # 6. Final Input Concatenation
        # (Batch, Seq, Embed*2) + (Batch, Seq, Embed) -> (Batch, Seq, Embed*3)
        inputs = torch.cat((context_expanded, embeddings), dim=2)
        
        # 7. Run Custom LSTM
        lstm_out, _ = self.lstm(inputs)
        
        # 8. Predict
        outputs = self.linear(lstm_out)
        
        return outputs