import torch.nn as nn
import torch
import torch.nn.functional as F


# class MF(nn.Module):
#     def __init__(self, num_user_embedding, num_item_embedding, embedding_dim):
#         super(MF, self).__init__()
#         self.user_embeddings = nn.Embedding(num_user_embedding, embedding_dim)
#         self.item_embeddings = nn.Embedding(num_item_embedding, embedding_dim)
#
#     def forward(self, x):
#         # x: (user_id, item_id), should be in shape (None, 2)
#         user_embedding = self.user_embeddings(x[:, 0])
#         item_embedding = self.item_embeddings(x[:, 1])
#         prediction = (user_embedding * item_embedding).sum(dim=1, keepdim=True)
#         return prediction

class MF(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=2):
        super(MF, self).__init__()
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        self.input_dim = embedding_dim * num_fields
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        # torch.nn.init.normal_(self.feature_embeddings.weight, mean=0, std=1e-4)

    def forward(self, feature_ids, feature_vals):
        # None*F*K -> None*(F*K)
        input_embeddings = self.feature_embeddings(feature_ids)
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        assert input_embeddings.shape[1] == 2
        output = torch.sum(input_embeddings[:, 0, :] * input_embeddings[:, 1, :], dim=1)
        return output.squeeze()


class MLP(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=2, hidden_size=128):
        super(MLP, self).__init__()
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        self.input_dim = embedding_dim * num_fields
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, feature_ids, feature_vals):
        # None*F*K -> None*(F*K)
        input_embeddings = self.feature_embeddings(feature_ids)
        assert input_embeddings.shape[1] == 2
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        input_embeddings = input_embeddings.view(-1, self.input_dim)
        output = nn.ReLU()(self.fc1(input_embeddings))
        output = nn.ReLU()(self.fc2(output))
        output = self.fc3(output)
        return output.squeeze()


class NeuMF(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=2, hidden_size=128):
        super(NeuMF, self).__init__()
        self.feature_embeddings_mlp = nn.Embedding(num_features, embedding_dim)
        self.feature_embeddings_gmf = nn.Embedding(num_features, embedding_dim)
        torch.nn.init.xavier_normal_(self.feature_embeddings_mlp.weight)
        torch.nn.init.xavier_normal_(self.feature_embeddings_gmf.weight)
        self.feature_embeddings = {"feature_embeddings_mlp": self.feature_embeddings_mlp,
                                   "feature_embeddings_gmf": self.feature_embeddings_gmf}
        self.input_dim = embedding_dim * num_fields
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2 + embedding_dim, 1)

    def forward(self, feature_ids, feature_vals):
        # None*F*K
        input_embeddings_gmf = self.feature_embeddings_gmf(feature_ids)
        assert input_embeddings_gmf.shape[1] == 2
        input_embeddings_gmf *= feature_vals.unsqueeze(dim=2)
        # None*K
        output_gmf = input_embeddings_gmf[:, 0, :] * input_embeddings_gmf[:, 1, :]
        # None*F*K -> None*(F*K)
        input_embeddings_mlp = self.feature_embeddings_mlp(feature_ids)
        input_embeddings_mlp *= feature_vals.unsqueeze(dim=2)
        output_mlp = input_embeddings_mlp.view(-1, self.input_dim)
        output_mlp = nn.ReLU()(self.fc1(output_mlp))
        output_mlp = nn.ReLU()(self.fc2(output_mlp))
        output = self.fc3(torch.cat([output_gmf, output_mlp], dim=1))
        return output.squeeze()


class FM(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=None):
        super(FM, self).__init__()
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        self.feature_biases = nn.Embedding(num_features, 1)
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        torch.nn.init.xavier_normal_(self.feature_biases.weight)
        self.bias = nn.Parameter(torch.zeros((1)))

    def forward(self, feature_ids, feature_vals):
        # None*F*K
        input_embeddings = self.feature_embeddings(feature_ids)
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        # None*K
        square_sum = torch.sum(input_embeddings ** 2, dim=1)
        sum_square = torch.sum(input_embeddings, dim=1) ** 2
        # None
        prediction = torch.mean((sum_square - square_sum) / 2, dim=1)
        # Add first order feature
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals
        prediction += torch.sum(input_biases, dim=1) + self.bias
        return prediction


class DeepFM(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=39, hidden_size=400):
        super(DeepFM, self).__init__()
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        self.feature_biases = nn.Embedding(num_features, 1)
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        torch.nn.init.xavier_normal_(self.feature_biases.weight)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.input_dim = embedding_dim * num_fields
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, feature_ids, feature_vals):
        # None*F*K
        input_embeddings = self.feature_embeddings(feature_ids)
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        # None*K
        square_sum = torch.sum(input_embeddings ** 2, dim=1)
        sum_square = torch.sum(input_embeddings, dim=1) ** 2
        # None
        prediction_fm = torch.mean((sum_square - square_sum) / 2, dim=1)
        # Add first order feature
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals
        prediction_fm += torch.sum(input_biases, dim=1) + self.bias

        input_embeddings_flatten = input_embeddings.view(-1, self.input_dim)
        hidden = nn.ReLU()(self.fc1(input_embeddings_flatten))
        hidden = nn.ReLU()(self.fc2(hidden))
        hidden = nn.ReLU()(self.fc3(hidden))
        prediction_dnn = self.fc4(hidden).squeeze()

        prediction = prediction_fm + prediction_dnn
        return prediction

class Wide_and_Deep(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields=39, hidden_size=400):
        super(Wide_and_Deep, self).__init__()
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        self.feature_biases = nn.Embedding(num_features, 1)
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        torch.nn.init.xavier_normal_(self.feature_biases.weight)
        self.bias = nn.Parameter(torch.zeros((1)))
        self.input_dim = embedding_dim * num_fields
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, feature_ids, feature_vals):
        # None*F*K
        input_embeddings = self.feature_embeddings(feature_ids)
        input_embeddings *= feature_vals.unsqueeze(dim=2)

        input_embeddings_flatten = input_embeddings.view(-1, self.input_dim)
        hidden = nn.ReLU()(self.fc1(input_embeddings_flatten))
        hidden = nn.ReLU()(self.fc2(hidden))
        hidden = nn.ReLU()(self.fc3(hidden))
        prediction_dnn = self.fc4(hidden).squeeze()

        # Add first order feature
        input_biases = self.feature_biases(feature_ids).squeeze()
        input_biases *= feature_vals
        prediction_lr = torch.sum(input_biases, dim=1) + self.bias

        prediction = prediction_dnn + prediction_lr
        return prediction


class xDeepFM(nn.Module):

    def __init__(self, num_features, embedding_dim, num_fields=39, cin_layer_size=(200, 200, 200),
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

        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        self.feature_biases = nn.Embedding(num_features, 1)
        self.input_dim = num_fields * embedding_dim
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
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

    def forward(self, feature_ids, feature_vals):
        inputs = self.feature_embeddings(feature_ids)
        inputs *= feature_vals.unsqueeze(dim=2)
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
