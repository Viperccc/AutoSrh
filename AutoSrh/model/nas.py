import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD



class AdamNas(Adam):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        # store param_groups_prime
        param_groups_prime = []
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for itr, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                #                 grad = p.grad.data
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    #                     grad.add_(group['weight_decay'], p.data)
                    grad.add_(group['weight_decay'], p)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                # store param_groups_prime
                param_groups_prime += [torch.addcdiv(torch.zeros_like(p), -step_size, exp_avg, denom)]
                state['exp_avg'] = exp_avg.detach().data
                state['exp_avg_sq'] = exp_avg_sq.detach().data
                p.data.addcdiv_(-step_size, exp_avg.data, denom.data)

        return loss, param_groups_prime


class SGDNas(SGD):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        param_groups_prime = []
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                #                 d_p = p.grad.data
                d_p = p.grad
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                param_groups_prime += [p - group['lr'] * d_p]
                p.data.add_(-group['lr'], d_p.data)

        return loss, param_groups_prime


class FM(nn.Module):
    def __init__(self, input_dim, num_features):
        super(FM, self).__init__()
        self.feature_biases = nn.Embedding(num_features, 1)
        torch.nn.init.zeros_(self.feature_biases.weight)
        self.bias = nn.Parameter(torch.zeros((1)), requires_grad=True)

    def forward(self, input_embeddings, feature_ids, feature_vals):
        # None*F*K -> None*K
        square_sum = torch.sum(input_embeddings ** 2, dim=1)
        sum_square = torch.sum(input_embeddings, dim=1) ** 2
        # -> None
        prediction = torch.mean((sum_square - square_sum) / 2, dim=1)
        # Add first order feature
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals
        prediction += torch.sum(input_biases, dim=1) + self.bias
        return prediction


class MLP(nn.Module):
    def __init__(self, input_dim, num_features=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, 1)

    def forward(self, input_embeddings, feature_ids=None, feature_vals=None):
        # None*F*K -> None*(F*K)
        input_embeddings = input_embeddings.view(-1, self.input_dim)
        output = nn.ReLU()(self.fc1(input_embeddings))
        output = nn.ReLU()(self.fc2(output))
        output = self.fc3(output)
        return output.squeeze()



class NeuMF(nn.Module):
    def __init__(self, input_dim, num_features=None, hidden_size=128):
        super(NeuMF, self).__init__()
        # 2*(2K)
        self.input_dim = input_dim//2
        self.d = self.input_dim//5
        self.fc1 = nn.Linear(8*self.d, 4*self.d)
        self.fc2 = nn.Linear(4*self.d, 2*self.d)
        self.fc3 = nn.Linear(2*self.d, self.d)
        self.fc4 = nn.Linear(2*self.d, 1)

    def forward(self, input_embeddings, feature_ids=None, feature_vals=None):
        # None*F*2K
        input_embeddings_gmf = input_embeddings[:, :, :self.d]
        assert input_embeddings_gmf.shape[1] == 2
        # None*K
        output_gmf = input_embeddings_gmf[:, 0, :] * input_embeddings_gmf[:, 1, :]
        # None*F*K -> None*(F*K)
        input_embeddings_mlp = input_embeddings[:, :, self.d:]
        #print(input_embeddings_mlp.shape)
        output_mlp = input_embeddings_mlp.reshape((-1, 8*self.d))
        #print(output_mlp.shape)
        output_mlp = nn.ReLU()(self.fc1(output_mlp))
        output_mlp = nn.ReLU()(self.fc2(output_mlp))
        output_mlp = nn.ReLU()(self.fc3(output_mlp))
        
        # print(input_embeddings.shape)
        # print(output_gmf.shape)
        # print(output_mlp.shape)
        output = self.fc4(torch.cat([output_gmf, output_mlp], dim=1))
        return output.squeeze()


class DeepFM(nn.Module):
    def __init__(self, input_dim, num_features=None, hidden_size=400):
        super(DeepFM, self).__init__()
        self.feature_biases = nn.Embedding(num_features, 1)
        torch.nn.init.xavier_normal_(self.feature_biases.weight)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, input_embeddings, feature_ids, feature_vals):
        # None*K
        square_sum = torch.sum(input_embeddings ** 2, dim=1)
        sum_square = torch.sum(input_embeddings, dim=1) ** 2
        # None
        prediction_fm = torch.mean((sum_square - square_sum) / 2, dim=1)
        # Add first order feature
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals
        prediction_fm += torch.sum(input_biases, dim=1) + self.bias
        #print(self.input_dim)
        #print(input_embeddings.size())
        input_embeddings_flatten = input_embeddings.view(-1, self.input_dim)
        hidden = nn.ReLU()(self.fc1(input_embeddings_flatten))
        hidden = nn.ReLU()(self.fc2(hidden))
        hidden = nn.ReLU()(self.fc3(hidden))
        prediction_dnn = self.fc4(hidden).squeeze()

        prediction = prediction_fm + prediction_dnn
        return prediction


class Wide_and_Deep(nn.Module):
    def __init__(self, input_dim, num_features=None, hidden_size=400):
        super(Wide_and_Deep, self).__init__()
        self.feature_biases = nn.Embedding(num_features, 1)
        torch.nn.init.xavier_normal_(self.feature_biases.weight)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, input_embeddings, feature_ids, feature_vals):
        # first order feature
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals
        prediction_lr = torch.sum(input_biases, dim=1) + self.bias

        # DNN
        input_embeddings_flatten = input_embeddings.view(-1, self.input_dim)
        hidden = nn.ReLU()(self.fc1(input_embeddings_flatten))
        hidden = nn.ReLU()(self.fc2(hidden))
        hidden = nn.ReLU()(self.fc3(hidden))
        prediction_dnn = self.fc4(hidden).squeeze()

        prediction = prediction_lr + prediction_dnn
        return prediction


class xDeepFM(nn.Module):

    def __init__(self, input_dim, num_features, num_fields=39, cin_layer_size=(200, 200, 200),
                 dnn_layer_size=(400, 400), activation=F.relu, split_half=True):
        super(xDeepFM, self).__init__()
        if len(cin_layer_size) == 0:
            raise ValueError(
                "cin_layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = cin_layer_size
        self.field_nums = [num_fields]
        self.split_half = split_half
        self.activation = activation
        self.dnn_layer_size = dnn_layer_size
        self.dnn_fcs = []

        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.feature_biases = nn.Embedding(num_features, 1)
        self.input_dim = input_dim
        torch.nn.init.xavier_normal_(self.feature_biases.weight)
        if self.split_half:
            hidden_size = sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]
        else:
            hidden_size = sum(self.layer_size)

        for i, size in enumerate(self.dnn_layer_size):
            if i == 0:
                self.dnn_fcs += [nn.Linear(self.input_dim, size)]
                last_size = size
            else:
                self.dnn_fcs += [nn.Linear(last_size, size)]
            self.dnn_fcs += [nn.ReLU()]
        self.dnn_net = nn.Sequential(*self.dnn_fcs)
        self.fc = nn.Linear(hidden_size + self.dnn_layer_size[-1], 1)

    def forward(self, inputs, feature_ids, feature_vals):
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals

        # CIN
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        cin_result = torch.cat(final_result, dim=1)
        cin_result = torch.sum(cin_result, -1)

        # DNN
        dnn_result = inputs.view(-1, self.input_dim)
        dnn_result = self.dnn_net(dnn_result)

        # Sum
        result = self.fc(torch.cat((cin_result, dnn_result), dim=1)).squeeze() + torch.sum(input_biases,
                                                                                           dim=1).squeeze()
        return result


class Dnis(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=2, model_name='FM',
                 feature_split=[0.1, 0.2, 0.2, 0.2, 0.3], num_dim_split=4,
                 search_space="block", normalize=True):
        super(Dnis, self).__init__()
        self.num_features = num_features
        self.feature_split = feature_split
        self.num_feature_split = len(feature_split)
        self.num_dim_split = num_dim_split
        self.search_space = search_space
        self.normalize = normalize
        if feature_split[0] > 1:
            self.feature_nums = torch.Tensor(feature_split).long()
        else:
            self.feature_nums = (torch.Tensor(feature_split) * num_features).long()
        self.feature_nums[-1] += num_features - self.feature_nums.sum()
        print(f"dnis.feature_nums:{self.feature_nums}")
        self.feature_positions = self._get_position(self.feature_nums)
        self.embed_dims = (torch.Tensor([1 / num_dim_split] * num_dim_split) * embedding_dim).long()
        self.embed_dims[-1] += embedding_dim - self.embed_dims.sum()
        # parameters in the embedding layer
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        # torch.nn.init.normal_(self.feature_embeddings.weight, mean=0, std=1e-4)
        if search_space == 'feature_block':
            self.alpha = nn.Parameter(
                torch.tensor(np.ones(num_dim_split, dtype=np.float32) * self.num_feature_split, requires_grad=True))
        elif search_space == 'embedding_block':
            self.alpha = nn.Parameter(
                torch.tensor(np.ones(self.num_feature_split, dtype=np.float32) * self.num_dim_split,
                             requires_grad=True))
        elif search_space == 'free':
            self.alpha = nn.Parameter(torch.tensor(
                np.ones([self.num_feature_split, num_dim_split], dtype=np.float32), requires_grad=True))
        else:
            raise NotImplementedError
        # projection matrix
        # self.projection = nn.Linear(embedding_dim, embedding_dim)
        # self.model = FM()
        self.model = globals()[model_name](input_dim=num_fields * embedding_dim, num_features=self.num_features)

    def forward(self, feature_ids, feature_vals):
        # feature_ids, feature_vals : None*F
        # single-side linear interpolation: 𝑊_𝑖𝑗=max⁡(min⁡(𝛼_𝑗−𝑖+1,1),0)
        if self.search_space == 'feature_block':
            alpha_block_weight = (self.alpha.expand(self.num_feature_split, self.num_dim_split) - torch.tensor(
                range(self.num_feature_split)).unsqueeze(1).float().to(self.alpha)).clamp(0, 1)
        elif self.search_space == 'embedding_block':
            alpha_block_weight = (
                    self.alpha.unsqueeze(1).expand(self.num_feature_split, self.num_dim_split) - torch.tensor(
                range(self.num_dim_split)).unsqueeze(0).float().to(self.alpha)).clamp(0, 1)
        elif self.search_space == 'free':
            alpha_block_weight = self.alpha
        else:
            raise NotImplementedError
        # #num_feature*K
        # alpha_block_mask = alpha_block_weight.repeat_interleave(self.feature_nums.to(self.alpha).long(),
        #                                                         dim=0).repeat_interleave(
        #     self.embed_dims.to(self.alpha).long(), dim=1)
        # #None*F*K
        # input_embedding_masks = alpha_block_mask[feature_ids]

        # None*F*K
  
        input_embeddings = self.feature_embeddings(feature_ids)
        # map feature_id to mask_id according to feature split positions, mask_id: None*F
        mask_id = ((feature_ids.unsqueeze(1) - self.feature_positions.to(feature_ids).unsqueeze(1).expand(
            self.feature_positions.shape[0], feature_ids.shape[1])) >= 0).sum(dim=1)
        # num_feature_split*K
        alpha_block_mask = alpha_block_weight.repeat_interleave(self.embed_dims.to(self.alpha).long(), dim=1)
        # None*F*K
        input_embedding_masks = alpha_block_mask[mask_id]
        # assert torch.all(input_embedding_masks==input_embedding_masks_new)
        if self.normalize:
            # None*F*1
            input_embedding_masks_sum = input_embedding_masks.sum(dim=2, keepdim=True)
            # normalize
            input_embedding_masks /= (input_embedding_masks_sum) + 1e-6
            input_embeddings *= self.num_dim_split
        input_embeddings *= input_embedding_masks
        # input_embeddings = self.projection(input_embeddings)
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        prediction = self.model(input_embeddings, feature_ids, feature_vals)
        return prediction

    def _get_position(self, feature_nums):
        position_list = []
        last = 0
        for e in feature_nums:
            position_list += [e + last]
            last += e
        return torch.Tensor(position_list).long()
