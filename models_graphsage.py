import torch
from torch import nn


class GraphSAGELayer(nn.Module):
    def __init__(self, inp, out,slope):
        super(GraphSAGELayer, self).__init__()
        self.W = nn.Linear(inp * 2, out)
        self.activation = nn.ReLU(slope)
    
    def forward(self, h, adj):
        N = adj.size(0)
        adj_sum = adj.sum(dim=1, keepdim=True)
        adj_normalized = adj.div(adj_sum)

        # performs matrix multiplication to aggregate information from neighboring nodes.
        aggregated = torch.mm(adj_normalized, h)
        # concatination of h and aggregated
        aggregated = torch.cat([h, aggregated], dim=1)
        h_hat = self.W(aggregated)
        h_hat = self.activation(h_hat)

        return h_hat

class MultiHeadGraphSAGE(nn.Module):
    def __init__(self, inp, out, heads,slope):
        super(MultiHeadGraphSAGE, self).__init__()
        # creates a list of GraphSAGELayer instances with the specified input size, output size, and slope for each attention head. 
        # The nn.ModuleList is used to store these instances as a list.
        self.attentions = nn.ModuleList([GraphSAGELayer(inp, out, slope) for _ in range(heads)])  # Ensure input and output sizes are consistent
        self.tanh = nn.Tanh()
    
    def forward(self, h, adj):
        # iterates over each attention head in self.attentions and applies it to the input features h and adjacency matrix adj
        heads_out = [att(h, adj) for att in self.attentions]
        # line stacks the outputs from all attention heads along a new dimension (dim=0) and then calculate the mean
        out = torch.stack(heads_out, dim=0).mean(0)
        
        return self.tanh(out)

class GraphSAGE(nn.Module):
    def __init__(self, inp, out, heads,slope=0.01):
        # to calls the constructor of the parent class nn.Module to properly initialize the MultiHeadGraphSAGE class.
        super(GraphSAGE, self).__init__()

        self.sage1 = MultiHeadGraphSAGE(inp, out, heads,slope)
        self.sage2 = MultiHeadGraphSAGE(out, out, heads,slope)
    
    def forward(self, h, adj):
        out = self.sage1(h, adj)
        out = self.sage2(out, adj)

        return out


# Using static embedding
class MAGNET(nn.Module):
  def __init__(self, input_size, hidden_size, adjacency, embeddings, heads=4, slope=0.01, dropout=0.5):
    super(MAGNET, self).__init__()

    self.embedding = nn.Embedding.from_pretrained(embeddings)

    self.rnn = nn.LSTM(input_size,
                        hidden_size,
                        batch_first=True,
                        bidirectional=True)

    self.graphsage = GraphSAGE(input_size, hidden_size*2, heads, slope)
    
    self.adjacency = nn.Parameter(adjacency)
    
    self.dropout = nn.Dropout(dropout)
 
  def forward(self, token, label_embedding):
    features = self.embedding(token)
    
    out, (hidden, cell) = self.rnn(features)
    
    out = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)
    out = self.dropout(out)
    
    att = self.dropout(self.graphsage(label_embedding, self.adjacency))
    att = att.transpose(0, 1)
    
    out = torch.mm(out, att)
 
    return out


#Contextual Embedding

class ContextMAGNET(nn.Module):
  def __init__(self, input_size, hidden_size, adjacency, heads=4, slope=0.01, dropout=0.5):
    super(ContextMAGNET, self).__init__()

    self.rnn = nn.LSTM(input_size,
                        hidden_size,
                        batch_first=True,
                        bidirectional=True)

    self.graphsage = GraphSAGE(input_size, hidden_size*2, heads, slope)
    
    self.adjacency = nn.Parameter(adjacency)
    
    self.dropout = nn.Dropout(dropout)
 
  def forward(self, features, label_embedding):

    out, (hidden, cell) = self.rnn(features)
    
    out = torch.cat([hidden[-2, :, :], hidden[-1, :, :]], dim=1)
    out = self.dropout(out)
    
    att = self.dropout(self.graphsage(label_embedding, self.adjacency))
    att = att.transpose(0, 1)
    
    out = torch.mm(out, att)
 
    return out