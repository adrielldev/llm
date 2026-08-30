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

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i,query)


attn_weights_2 = torch.softmax(attn_scores_2,dim=0)

context_vec_2 = torch.zeros(query.shape)
#print(context_vec_2)
#print(attn_weights_2)
for i,x_i in enumerate(inputs):
    print(i)
    print(x_i)
    print(attn_weights_2[i])
    context_vec_2 += attn_weights_2[i]*x_i
    print(context_vec_2)
#print(context_vec_2)