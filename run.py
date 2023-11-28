import pickle
import argparse

from model.Discriminators import Discriminator
from model.CD import NCDM, IRT, MIRT
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
import os
import time

args = argparse.ArgumentParser()
args.add_argument("-DATA", default="pisa2015", type=str)
args.add_argument("-FILTER_MODE", default="separate", type=str)
args.add_argument(
    "-FEATURES",
    default=["OECD", "GENDER", "EDU", "ECONOMIC"],
    type=list,
)  # ["OECD", "GENDER", "EDU", "ECONOMIC"]
args.add_argument("-FAIRNESS_RATIO", default=1.0, type=float)
args.add_argument("-FAIRNESS_RATIO_NOFEATURE", default=0.5, type=float)
args.add_argument("-CUDA", default=2, type=int)
args.add_argument("-SEED", default=4869, type=int)
args.add_argument("-BATCH_SIZE", default=8192, type=int)
args.add_argument("-EPOCH", default=10, type=int)
args.add_argument("-EPOCH_DISCRIMINATOR", default=10, type=int)
args.add_argument("-EPOCH_ATTACKER", default=10, type=int)
args.add_argument("-USER_NUM", default=358415, type=int)
args.add_argument("-ITEM_NUM", default=183, type=int)
args.add_argument("-KNOWLEDGE_NUM", default=16, type=int)
args.add_argument("-LATENT_NUM", default=16, type=int)
args.add_argument("-MODEL", default="NCDM", type=str)
args.add_argument("-LR", default=0.001, type=float)
args.add_argument("-LR_DISC", default=0.01, type=float)
args.add_argument("-USE_NOFEATURE", default=False, type=bool)
args.add_argument("-NO_FEATURE", default=0.2, type=float)
args = args.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA)
torch.manual_seed(args.SEED)
random.seed(args.SEED)
np.random.seed(args.SEED)
model_name = args.DATA + args.MODEL + args.FILTER_MODE + str(args.FAIRNESS_RATIO)
print(args)
print(model_name)


def transform(user, item, score, feature, nofeature=False):
    if nofeature == False:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(feature), dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            torch.tensor(np.array(score), dtype=torch.float32),
        )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


def transform_know(user, item, item2knowledge, score, feature, nofeature=False):
    knowledge_emb = item2knowledge[list(item)]
    if nofeature == False:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            knowledge_emb,
            torch.tensor(np.array(score), dtype=torch.float32),
            torch.tensor(np.array(feature), dtype=torch.float32),
        )
    else:
        dataset = TensorDataset(
            torch.tensor(np.array(user), dtype=torch.int64),
            torch.tensor(np.array(item), dtype=torch.int64),
            knowledge_emb,
            torch.tensor(np.array(score), dtype=torch.float32),
        )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


def attacker_transform(user, feature):
    dataset = TensorDataset(
        torch.tensor(np.array(user), dtype=torch.int64),
        torch.tensor(np.array(feature), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=True)


print("load data")

pkl = open("./data/" + args.DATA + "/item2knowledge.pkl", "rb")
item2knowledge = pickle.load(pkl)
pkl.close()

train_data = pd.read_csv("./data/" + args.DATA + "/pisa.train.csv")
valid_data = pd.read_csv("./data/" + args.DATA + "/pisa.validation.csv")
test_data = pd.read_csv("./data/" + args.DATA + "/pisa.test.csv")
train_data_nofeature = train_data[
    train_data["user_id"] <= int(args.USER_NUM * args.NO_FEATURE)
]
train_data = train_data[train_data["user_id"] > int(args.USER_NUM * args.NO_FEATURE)]

test_data_group = []
for feature in args.FEATURES:
    test_data_group.append(test_data[test_data[feature] == 0])
    test_data_group.append(test_data[test_data[feature] == 1])

attacker_train_data = pd.read_csv("./data/" + args.DATA + "/pisa.attacker.train.csv")
attacker_test_data = pd.read_csv("./data/" + args.DATA + "/pisa.attacker.test.csv")

attacker_train, attacker_test = [
    attacker_transform(data["user_id"], data[args.FEATURES])
    for data in [attacker_train_data, attacker_test_data]
]

if args.MODEL in ["MCD", "IRT", "MIRT"]:
    train, train_nofeature, valid, test = [
        transform(
            data["user_id"],
            data["item_id"],
            data["score"],
            data[args.FEATURES],
        )
        for data in [train_data, train_data_nofeature, valid_data, test_data]
    ]
    test_group = [
        transform(
            data["user_id"],
            data["item_id"],
            data["score"],
            data[args.FEATURES],
        )
        for data in test_data_group
    ]
elif args.MODEL in ["NCDM"]:
    train, train_nofeature, valid, test = [
        transform_know(
            data["user_id"],
            data["item_id"],
            item2knowledge,
            data["score"],
            data[args.FEATURES],
        )
        for data in [train_data, train_data_nofeature, valid_data, test_data]
    ]
    test_group = [
        transform_know(
            data["user_id"],
            data["item_id"],
            item2knowledge,
            data["score"],
            data[args.FEATURES],
        )
        for data in test_data_group
    ]
train_total = [train_nofeature, train]


print("train model")

cdm = eval(args.MODEL)(args)
trainer = torch.optim.Adam(cdm.parameters(), args.LR)
cdm.train()

if args.MODEL == "IRT":
    discriminators = {}
    for feature in args.FEATURES:
        discriminators[feature] = Discriminator(args, 1).cuda()
        discriminators[feature].train()
    discriminator_trainers = {
        feature: torch.optim.Adam(discriminators[feature].parameters(), args.LR_DISC)
        for feature in args.FEATURES
    }
if args.MODEL == "MIRT":
    discriminators = {}
    for feature in args.FEATURES:
        discriminators[feature] = Discriminator(args, args.LATENT_NUM).cuda()
        discriminators[feature].train()
    discriminator_trainers = {
        feature: torch.optim.Adam(discriminators[feature].parameters(), args.LR_DISC)
        for feature in args.FEATURES
    }
if args.MODEL == "NCDM":
    discriminators = {}
    for feature in args.FEATURES:
        discriminators[feature] = Discriminator(args, args.KNOWLEDGE_NUM).cuda()
        discriminators[feature].train()
    discriminator_trainers = {
        feature: torch.optim.Adam(discriminators[feature].parameters(), args.LR_DISC)
        for feature in args.FEATURES
    }

for epoch in range(args.EPOCH):
    for whether_nofeature, train_iter in enumerate(train_total):
        for batch_data in tqdm(train_iter, "Epoch %s" % epoch):
            if args.MODEL in ["MCD", "IRT", "MIRT"]:
                user_id, item_id, response, feature_data = batch_data
            else:
                user_id, item_id, knowledge, response, feature_data = batch_data
                knowledge = knowledge.to("cuda")
            user_id = user_id.to("cuda")
            item_id = item_id.to("cuda")
            response = response.to("cuda")
            feature_data = feature_data.to("cuda")
            mask = np.random.randint(len(args.FEATURES))
            if args.MODEL in ["MCD", "IRT", "MIRT"]:
                out = cdm(user_id, item_id, response, mask)
            elif args.MODEL in ["NCDM"]:
                out = cdm(user_id, item_id, knowledge, response, mask)
            trainer.zero_grad()
            penalty = 0
            if args.USE_NOFEATURE or whether_nofeature == 1:
                for i, feature in enumerate(args.FEATURES):
                    if args.FILTER_MODE == "combine":
                        if mask != None:
                            if i == mask:
                                if whether_nofeature == 1:
                                    penalty -= args.FAIRNESS_RATIO * discriminators[
                                        feature
                                    ](out["u_vector"][0], feature_data[:, i])
                                else:
                                    penalty -= (
                                        args.FAIRNESS_RATIO_NOFEATURE
                                        * args.FAIRNESS_RATIO
                                        * discriminators[feature].hforward(
                                            out["u_vector"][0]
                                        )
                                    )
                        else:
                            if whether_nofeature == 1:
                                penalty -= args.FAIRNESS_RATIO * discriminators[
                                    feature
                                ](out["u_vector"][i], feature_data[:, i])
                            else:
                                penalty -= (
                                    args.FAIRNESS_RATIO_NOFEATURE
                                    * args.FAIRNESS_RATIO
                                    * discriminators[feature].hforward(
                                        out["u_vector"][i]
                                    )
                                )
                    else:
                        if whether_nofeature == 1:
                            penalty -= args.FAIRNESS_RATIO * discriminators[feature](
                                out["u_vector"][i], feature_data[:, i]
                            )
                        else:
                            penalty -= (
                                args.FAIRNESS_RATIO_NOFEATURE
                                * args.FAIRNESS_RATIO
                                * discriminators[feature].hforward(
                                    out["u_vector"][i]
                                )
                            )

            loss = out["loss"] + penalty
            loss.backward()
            trainer.step()

            if args.USE_NOFEATURE or whether_nofeature == 1:
                for _ in range(args.EPOCH_DISCRIMINATOR):
                    for i, feature in enumerate(args.FEATURES):
                        if args.FILTER_MODE == "combine":
                            if i == mask:
                                if whether_nofeature == 1:
                                    discriminator_trainers[feature].zero_grad()
                                    disc_loss = args.FAIRNESS_RATIO * discriminators[
                                        feature
                                    ](out["u_vector"][0].detach(), feature_data[:, i])
                                    disc_loss.backward()
                                    discriminator_trainers[feature].step()
                        else:
                            if whether_nofeature == 1:
                                discriminator_trainers[feature].zero_grad()
                                disc_loss = args.FAIRNESS_RATIO * discriminators[
                                    feature
                                ](out["u_vector"][i].detach(), feature_data[:, i])
                                disc_loss.backward()
                                discriminator_trainers[feature].step()

    cdm.eval()
    y_pred = []
    y_true = []
    for batch_data in tqdm(test, "Test"):
        if args.MODEL in ["MCD", "IRT", "MIRT"]:
            user_id, item_id, response, _ = batch_data
        elif args.MODEL in ["NCDM"]:
            user_id, item_id, knowledge, response, _ = batch_data
            knowledge = knowledge.to("cuda")
        user_id = user_id.to("cuda")
        item_id = item_id.to("cuda")
        response = response.to("cuda")
        if args.MODEL in ["MCD", "IRT", "MIRT"]:
            out = cdm.predict(user_id, item_id)
        elif args.MODEL in ["NCDM"]:
            out = cdm.predict(user_id, item_id, knowledge)
        y_pred.extend(out["prediction"].tolist())
        y_true.extend(response.tolist())
    print("eval:{:d}".format(epoch))
    print("acc:{:.4f}".format(accuracy_score(y_true, np.array(y_pred) > 0.5)))
    print("auc:{:.4f}".format(roc_auc_score(y_true, y_pred)))
    print("mae:{:.4f}".format(mean_absolute_error(y_true, y_pred)))
    print("mse:{:.4f}".format(mean_squared_error(y_true, y_pred)))

    cdm.train()

    if args.MODEL == "IRT":
        attackers = {}
        for feature in args.FEATURES:
            attackers[feature] = Discriminator(args, 1)
            attackers[feature].train()
        attacker_trainers = {
            feature: torch.optim.Adam(attackers[feature].parameters(), args.LR_DISC)
            for feature in args.FEATURES
        }
    if args.MODEL == "MIRT":
        attackers = {}
        for feature in args.FEATURES:
            attackers[feature] = Discriminator(args, args.LATENT_NUM)
            attackers[feature].train()
        attacker_trainers = {
            feature: torch.optim.Adam(attackers[feature].parameters(), args.LR_DISC)
            for feature in args.FEATURES
        }
    if args.MODEL == "NCDM":
        attackers = {}
        for feature in args.FEATURES:
            attackers[feature] = Discriminator(args, args.KNOWLEDGE_NUM)
            attackers[feature].train()
        attacker_trainers = {
            feature: torch.optim.Adam(attackers[feature].parameters(), args.LR_DISC)
            for feature in args.FEATURES
        }

    u_embeddings = cdm.apply_filter(cdm.filter_u_dict, cdm.theta.weight).detach()
    u_embeddings = torch.mean(u_embeddings, dim=0).detach()

    best_result = {}
    best_epoch = {}
    for feature in args.FEATURES:
        best_result[feature] = 0
        best_epoch[feature] = 0
    for epoch in range(args.EPOCH_ATTACKER):
        for batch_data in tqdm(attacker_train, "Epoch %s" % epoch):
            user_id, feature_label = batch_data
            user_id = user_id.to("cuda")
            for i, feature in enumerate(args.FEATURES):
                label = feature_label[:, i].to("cuda")
                u_embedding = u_embeddings[user_id, :]
                attacker_trainers[feature].zero_grad()
                loss = attackers[feature](u_embedding, label)
                loss.backward()
                attacker_trainers[feature].step()

        feature_pred = {}
        feature_true = {}
        for feature in args.FEATURES:
            feature_pred[feature] = []
            feature_true[feature] = []
            attackers[feature].eval()
        for batch_data in tqdm(attacker_test, "Test"):
            user_id, feature_label = batch_data
            user_id = user_id.to("cuda")
            for i, feature in enumerate(args.FEATURES):
                u_embedding = u_embeddings[user_id, :]
                pred = attackers[feature].predict(u_embedding).cpu()
                feature_pred[feature].extend(pred.tolist())
                feature_true[feature].extend(feature_label[:, i].tolist())
        for feature in args.FEATURES:
            print(feature)
            print("auc:", roc_auc_score(feature_true[feature], feature_pred[feature]))
        for feature in args.FEATURES:
            attackers[feature].train()

    for feature in args.FEATURES:
        print(feature)
        print("auc:", roc_auc_score(feature_true[feature], feature_pred[feature]))

print("finish")
