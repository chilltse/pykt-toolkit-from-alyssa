#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor


def calculate_sgap_seq(concepts, max_gap=300):
    """
    For each position in a single sequence, calculate gap to NEXT occurrence of same concept.

    Args:
        concepts: List of concept IDs for one sequence
        max_gap: Maximum gap value to cap at (default 300)

    Returns:
        sgap: List of gap values with same length as concepts
    """
    seq_len = len(concepts)
    sgap = [max_gap - 1] * seq_len  # Default: concept never occurs again

    for i in range(seq_len):
        concept = concepts[i]
        if concept == -1:  # Skip padding
            sgap[i] = 0
            continue
        # Find next occurrence of same concept
        for j in range(i + 1, seq_len):
            if concepts[j] == concept:
                sgap[i] = min(j - i, max_gap - 1)
                break

    return sgap


def calculate_pcount_seq(concepts):
    """
    For each position in a single sequence, count items since LAST occurrence of same concept.

    Args:
        concepts: List of concept IDs for one sequence

    Returns:
        pcount: List of count values with same length as concepts
    """
    seq_len = len(concepts)
    pcount = [0] * seq_len
    concept_last_pos = {}

    for i in range(seq_len):
        concept = concepts[i]
        if concept == -1:  # Skip padding
            pcount[i] = 0
            continue
        if concept in concept_last_pos:
            pcount[i] = i - concept_last_pos[concept]
        else:
            pcount[i] = 0  # First occurrence
        concept_last_pos[concept] = i

    return pcount


class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            # Use new suffix to avoid loading old cached data without dgaps
            processed_data = file_path + folds_str + "_qtest_v2.pkl"
        else:
            processed_data = file_path + folds_str + "_v2.pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dgaps, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dgaps, self.dqtest]
            else:
                self.dori, self.dgaps = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dgaps]
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dgaps, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori, self.dgaps = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        Args:
            index (int): the index of the data want to get
        Returns:
            (tuple): tuple containing:

            - **dcur (dict)**: dictionary with sequence data:
                - q_seqs (torch.tensor): question id sequence of the 0~seqlen-2 interactions
                - c_seqs (torch.tensor): knowledge concept id sequence of the 0~seqlen-2 interactions
                - r_seqs (torch.tensor): response id sequence of the 0~seqlen-2 interactions
                - qshft_seqs (torch.tensor): question id sequence of the 1~seqlen-1 interactions
                - cshft_seqs (torch.tensor): knowledge concept id sequence of the 1~seqlen-1 interactions
                - rshft_seqs (torch.tensor): response id sequence of the 1~seqlen-1 interactions
                - mask_seqs (torch.tensor): masked value sequence, shape is seqlen-1
                - select_masks (torch.tensor): is select to calculate the performance or not
            - **dcurgaps (dict)**: dictionary with interference metrics:
                - rgaps (torch.tensor): time gap to last occurrence of same concept (minutes, based on timestamps)
                - sgaps (torch.tensor): time gap to previous item (minutes, based on timestamps)
                - pcounts (torch.tensor): cumulative count of concept occurrences before current position
                - pcumcounts (torch.tensor): count of items since last occurrence of same concept (position-based)
            - **dqtest (dict)**: used only self.qtest is True, for question level evaluation
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_"+key] = self.dori[key]
                continue
            # print(f"key: {key}, len: {len(self.dori[key])}")
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_"+key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

        # Process interference gap data (sgaps, pcounts)
        dcurgaps = dict()
        for key in self.dgaps:
            seqs = self.dgaps[key][index][:-1] * mseqs
            shft_seqs = self.dgaps[key][index][1:] * mseqs
            dcurgaps[key] = seqs
            dcurgaps["shft_"+key] = shft_seqs

        if not self.qtest:
            return dcur, dcurgaps
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dcurgaps, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        Args:
            sequence_path (str): file path of the sequences
            folds (list[int]):
            pad_val (int, optional): pad value. Defaults to -1.
        Returns:
            (tuple): tuple containing
            - **dori (dict)**: dictionary with sequence tensors
            - **dgaps (dict)**: dictionary with interference metrics (sgaps, pcounts)
            - **dqtest (dict)**: not null only self.qtest is True, for question level evaluation
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        # Interference gap data for AKT interference-based forgetting
        # 使用类似dkt_forget的指标：rgap, sgap, pcount（基于时间戳）
        # 以及pcumcount（位置间隔）
        dgaps = {"rgaps": [], "sgaps": [], "pcounts": [], "pcumcounts": [], "scumcounts": []}

        df = pd.read_csv(sequence_path)#[0:1000]
        df = df[df["fold"].isin(folds)]
        interaction_num = 0
        dqtest = {"qidxs": [], "rests":[], "orirow":[]}
        for i, row in df.iterrows():
            #use kc_id or question_id as input
            if "concepts" in self.input_type:
                concepts = [int(_) for _ in row["concepts"].split(",")]
                dori["cseqs"].append(concepts)
            else:
                concepts = None
            if "questions" in self.input_type:
                questions = [int(_) for _ in row["questions"].split(",")]
                dori["qseqs"].append(questions)
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            # Calculate interference metrics
            # 1. 基于时间戳的指标（rgap, sgap, pcount）- 使用calC函数
            if "timestamps" in row:
                rgap, sgap, pcount = self.calC(row)
            else:
                # 如果没有时间戳，使用零值
                seq_len = len(dori["rseqs"][-1])
                rgap = [0] * seq_len
                sgap = [0] * seq_len
                pcount = [0] * seq_len
            
            # 2. 基于位置的指标（pcumcount）- 自上次相同概念出现以来的项目数
            if concepts is not None:
                pcumcount = calculate_pcount_seq(concepts)
                scumcount = calculate_sgap_seq(concepts)  # 添加：到下一个相同概念出现的距离
            elif "questions" in self.input_type:
                pcumcount = calculate_pcount_seq(questions)
                scumcount = calculate_sgap_seq(questions)  # 添加：到下一个相同概念出现的距离
            else:
                # Fallback: create zero arrays
                seq_len = len(dori["rseqs"][-1])
                pcumcount = [0] * seq_len
                scumcount = [0] * seq_len  # 添加
            
            dgaps["rgaps"].append(rgap)
            dgaps["sgaps"].append(sgap)
            dgaps["pcounts"].append(pcount)
            dgaps["pcumcounts"].append(pcumcount)
            dgaps["scumcounts"].append(scumcount)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])
        for key in dori:
            if key not in ["rseqs"]:#in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        # Convert dgaps to tensors
        for key in dgaps:
            dgaps[key] = LongTensor(dgaps[key])

        mask_seqs = (dori["cseqs"][:,:-1] != pad_val) * (dori["cseqs"][:,1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]

            return dori, dgaps, dqtest
        return dori, dgaps

    def log2(self, t):
        """计算log2(t+1)，用于时间间隔的归一化"""
        import math
        return round(math.log(t+1, 2))

    def calC(self, row):
        """
        计算基于时间戳的干扰指标（类似dkt_forget）
        
        返回:
            repeated_gap (rgap): 到上次相同概念出现的时间间隔（分钟）
            sequence_gap (sgap): 到上一个项目的时间间隔（分钟）
            past_counts (pcount): 该概念之前出现的累积次数
        """
        repeated_gap, sequence_gap, past_counts = [], [], []
        uid = row["uid"]
        # default: concepts
        skills = row["concepts"].split(",") if "concepts" in self.input_type else row["questions"].split(",")
        timestamps = row["timestamps"].split(",")
        dlastskill, dcount = dict(), dict()
        pret = None
        for s, t in zip(skills, timestamps):
            s, t = int(s), int(t)
            if s not in dlastskill or s == -1:
                curRepeatedGap = 0
            else:
                curRepeatedGap = self.log2((t - dlastskill[s]) / 1000 / 60) + 1  # minutes
            dlastskill[s] = t

            repeated_gap.append(curRepeatedGap)
            if pret == None or t == -1:
                curLastGap = 0
            else:
                curLastGap = self.log2((t - pret) / 1000 / 60) + 1
            pret = t
            sequence_gap.append(curLastGap)

            dcount.setdefault(s, 0)
            past_counts.append(self.log2(dcount[s]))
            dcount[s] += 1
        return repeated_gap, sequence_gap, past_counts