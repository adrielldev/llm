import torch

inputs = torch.tensor(
    [[0.43,0.15,0.89],
    [0.55,0.87,0.66],
    [0.57,0.85,0.64],
    [0.22,0.58,0.33],
    [0.77,0.25,0.10],
    [0.05,0.8,0.55]
    ]
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)

w_query = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
w_key = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)
w_value = torch.nn.Parameter(torch.rand(d_in,d_out),requires_grad=False)

query_2 = x_2 @ w_query
key_2 = x_2 @ w_key
value_2 = x_2 @ w_value

keys = inputs @ w_key
values = inputs @ w_value

keys_2 = keys[1]
attn_scores_2 = query_2 @ keys.T


d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5,dim=-1)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)