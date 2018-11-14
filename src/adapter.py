'''
This is a pytorch implementation of the MLP Adapter for each
lateral connection in the Progressive NN.

The governing equation for this network looks like:
    U_i^{k:j} \sigma(V_i^{} \alpha_{i-1}^{<k} h_{i-1}^{<k}
    U is the lateral connections, which gets multiplied by the 
    MLP to project it down into a n_{i-1} size subspace
'''

# There should be a check in here for cuda
# ex if (torch.cuda.is_available) do blah

class Adapter(nn.Module):
    def __init__(self, h_sizes, out_size):

        super().__init__()
        
        # Make sure its just a single hidden layer for now
        assert len(h_sizes) == 2

        # Input and single hidden layer
        self.hidden = nn.ModuleList() 
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Output layer 
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        
        output = F.relu(self.out(x), dim=1)
        
        return output
