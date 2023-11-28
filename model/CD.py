import torch
import torch.nn as nn

class BaseCD(nn.Module):
    def __init__(self, args):
        super(BaseCD, self).__init__()
        self.user_num = args.USER_NUM
        self.item_num = args.ITEM_NUM
        self.knowledge_num = args.KNOWLEDGE_NUM
        self.num_features = len(args.FEATURES)
        self.filter_mode = args.FILTER_MODE
        self.criterion = nn.BCELoss()

    def get_sensitive_filter(self, embed_dim):
        sequential = nn.Sequential(
            nn.Linear(embed_dim, self.knowledge_num),
            nn.ReLU(),
            nn.Linear(self.knowledge_num, self.knowledge_num),
            nn.ReLU(),
            nn.Linear(self.knowledge_num, embed_dim),
        ).to("cuda")
        return sequential

    def apply_filter(self, filter_dict, vectors, mask=None):
        if self.num_features != 0:
            if self.filter_mode == "separate":
                return torch.stack(
                    [filter_dict["1"](vectors) for _ in range(self.num_features)], 0
                )
            elif self.filter_mode == "combine":
                result = []
                if mask == None:
                    for i in range(self.num_features):
                        result.append(filter_dict[str(i + 1)](vectors))
                else:
                    result.append(filter_dict[str(mask + 1)](vectors))
                return torch.stack(result, 0)
            else:
                assert "error!"
        else:
            return vectors

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))


class IRT(BaseCD):
    def __init__(self, args):
        super(IRT, self).__init__(args)
        if self.filter_mode == "combine":
            self.filter_u_dict = nn.ModuleDict(
                {
                    str(i + 1): self.get_sensitive_filter(1)
                    for i in range(self.num_features)
                }
            )
        elif self.filter_mode == "separate":
            self.filter_u_dict = nn.ModuleDict(
                {str(i + 1): self.get_sensitive_filter(1) for i in range(1)}
            )
        else:
            assert "error!"
        self.theta = nn.Embedding(self.user_num, 1).to("cuda")
        self.a = nn.Embedding(self.item_num, 1).to("cuda")
        self.b = nn.Embedding(self.item_num, 1).to("cuda")
        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)

    def predict(self, user_id, item_id, mask=None):
        thetas = self.apply_filter(self.filter_u_dict, self.theta(user_id), mask=mask)
        alpha = self.a(item_id)
        beta = self.b(item_id)
        theta = torch.mean(thetas, dim=0)
        pred = alpha * (theta - beta)
        pred = torch.squeeze(torch.sigmoid(pred), 1)
        out = {"prediction": pred}
        out["u_vector"] = thetas
        return out

    def forward(self, user_id, item_id, score, mask=None):
        out = self.predict(user_id, item_id, mask=mask)
        loss = self.criterion(out["prediction"], score)
        out["loss"] = loss
        return out


class MIRT(BaseCD):
    def __init__(self, args):
        super(MIRT, self).__init__(args)
        if self.filter_mode == "combine":
            self.filter_u_dict = nn.ModuleDict(
                {
                    str(i + 1): self.get_sensitive_filter(args.LATENT_NUM)
                    for i in range(self.num_features)
                }
            )
        elif self.filter_mode == "separate":
            self.filter_u_dict = nn.ModuleDict(
                {
                    str(i + 1): self.get_sensitive_filter(args.LATENT_NUM)
                    for i in range(1)
                }
            )
        else:
            assert "error!"
        self.theta = nn.Embedding(self.user_num, args.LATENT_NUM).to("cuda")
        self.a = nn.Embedding(self.item_num, args.LATENT_NUM).to("cuda")
        self.b = nn.Embedding(self.item_num, 1).to("cuda")
        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)

    def predict(self, user_id, item_id, mask=None):
        thetas = self.apply_filter(self.filter_u_dict, self.theta(user_id), mask=mask)
        alpha = self.a(item_id)
        beta = self.b(item_id)
        theta = torch.mean(thetas, dim=0)
        pred = torch.sum(alpha * theta, dim=1).unsqueeze(1) - beta
        pred = torch.squeeze(torch.sigmoid(pred), 1)
        out = {"prediction": pred}
        out["u_vector"] = thetas
        return out

    def forward(self, user_id, item_id, score, mask=None):
        out = self.predict(user_id, item_id, mask=mask)
        loss = self.criterion(out["prediction"], score)
        out["loss"] = loss
        return out

class NCDM(BaseCD):
    def __init__(self, args):
        super(NCDM, self).__init__(args)
        self.knowledge_dim = args.KNOWLEDGE_NUM
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        self.theta = nn.Embedding(self.user_num, self.knowledge_dim).to("cuda")
        self.k_difficulty = nn.Embedding(self.item_num, self.knowledge_dim).to("cuda")
        self.e_difficulty = nn.Embedding(self.item_num, 1).to("cuda")
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1).to(
            "cuda"
        )
        self.drop_1 = nn.Dropout(p=0.5).to("cuda")
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2).to("cuda")
        self.drop_2 = nn.Dropout(p=0.5).to("cuda")
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1).to("cuda")
        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.xavier_uniform_(self.k_difficulty.weight)
        nn.init.xavier_uniform_(self.e_difficulty.weight)
        nn.init.xavier_uniform_(self.prednet_full1.weight)
        nn.init.xavier_uniform_(self.prednet_full2.weight)
        nn.init.xavier_uniform_(self.prednet_full3.weight)
        if self.filter_mode == "combine":
            self.filter_u_dict = nn.ModuleDict(
                {
                    str(i + 1): self.get_sensitive_filter(self.knowledge_dim)
                    for i in range(self.num_features)
                }
            )
        elif self.filter_mode == "separate":
            self.filter_u_dict = nn.ModuleDict(
                {
                    str(i + 1): self.get_sensitive_filter(self.knowledge_dim)
                    for i in range(1)
                }
            )
        else:
            assert "error!"

    def predict(self, user_id, item_id, input_knowledge_point, mask=None):
        thetas = self.apply_filter(self.filter_u_dict, self.theta(user_id), mask=mask)
        theta = torch.mean(thetas, dim=0)
        stat_emb = torch.sigmoid(theta)
        k_vector = self.k_difficulty(item_id)
        e_vector = self.e_difficulty(item_id)
        k_difficulty = torch.sigmoid(k_vector)
        e_difficulty = torch.sigmoid(e_vector)
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x)).view(-1)
        out = {"prediction": output_1}
        out["u_vector"] = thetas
        return out

    def forward(self, user_id, item_id, input_knowledge_point, score, mask=None):
        out = self.predict(user_id, item_id, input_knowledge_point, mask=mask)
        loss = self.criterion(out["prediction"], score)
        out["loss"] = loss
        return out
