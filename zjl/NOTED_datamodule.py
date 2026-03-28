from typing import Optional, Sequence, Union

import lamindb as ln
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import (
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

from .collator import Collator
from .data import Dataset
from .utils import getBiomartTable


class DataModule(L.LightningDataModule): #DataModule 类继承自 L.LightningDataModule，用于管理数据集的加载、分割和处理。
    def __init__(
        self,
        collection_name: str, #要使用的 lamindb 集合。
        clss_to_weight: list = ["organism_ontology_term_id"], #在训练器的加权随机取样器中要加权的类。
        organisms: list = ["NCBITaxon:9606"], # 数据集中要包含的生物。
        weight_scaler: int = 10, #最多出现与较少出现类别的权重
        train_oversampling_per_epoch: float = 0.1, #每个 epoch 的训练集中包含的数据集比例。
        validation_split: float = 0.2, #在验证分割中包含的数据集比例
        test_split: float = 0, #要包含在测试分割中的数据集比例
        gene_embeddings: str = "", # 基因嵌入文件的路径。
        use_default_col: bool = True, # 是否使用默认整理器。
        gene_position_tolerance: int = 10_000, #基因位置的容差
        # this is for the mappedCollection
        all_clss: list = ["organism_ontology_term_id"], #所有类别的列表
        hierarchical_clss: list = [], #分层类别列表。
        # this is for the collator
        how: str = "random expr", #整理器使用的方法。默认为 “随机 expr”。
        organism_name: str = "organism_ontology_term_id",
        max_len: int = 1000, #输入张量的最大长度
        add_zero_genes: int = 100,
        do_gene_pos: Union[bool, str] = True, #是否使用基因位置
        tp_name: Optional[str] = None,  # "heat_diff" #时间点名称。
        assays_to_drop: list = [
            "EFO:0008853",
            "EFO:0010961",
            "EFO:0030007",
            "EFO:0030062",
        ],
        **kwargs,
    ):
        """
        DataModule a pytorch lighting datamodule directly from a lamin Collection.  DataModule 直接从 lamin Collection 获取 pytorch lighting datamodule
        it can work with bare pytorch too 也可用于裸 pytorch
        该类的主要功能是将数据集划分为训练集、验证集和测试集，并为每个集创建相应的数据加载器（DataLoader）。
        It implements train / val / test dataloaders. the train is weighted random, val is random, test is one to many separated datasets.  它实现了 train / val / test 数据加载器。train 是加权随机数据，val 是随机数据，test 是一到多个分离的数据集。
        This is where the mappedCollection, dataset, and collator are combined to create the dataloaders. 在这里，mappedCollection、数据集和整理器被组合起来，以创建数据载体。
      
        Args:
            collection_name (str): The lamindb collection to be used.
            clss_to_weight (list, optional): The classes to weight in the trainer's weighted random sampler. Defaults to ["organism_ontology_term_id"].
            organisms (list, optional): The organisms to include in the dataset. Defaults to ["NCBITaxon:9606"].
            weight_scaler (int, optional): how much more you will see the most present vs less present category.
            train_oversampling_per_epoch (float, optional): The proportion of the dataset to include in the training set for each epoch. Defaults to 0.1.
            validation_split (float, optional): The proportion of the dataset to include in the validation split. Defaults to 0.2.
            test_split (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.
                it will use a full dataset and will round to the nearest dataset's cell count. 使用完整数据集，并四舍五入到最接近的数据集单元格数。
            gene_embeddings (str, optional): The path to the gene embeddings file. Defaults to "".
                the file must have ensembl_gene_id as index. 该文件必须有 ensembl_gene_id 作为索引。
                This is used to subset the available genes further to the ones that have embeddings in your model. 这用于将可用基因进一步细分为模型中有嵌入的基因。
            use_default_col (bool, optional): Whether to use the default collator. Defaults to True.
            gene_position_tolerance (int, optional): The tolerance for gene position. Defaults to 10_000.
                any genes within this distance of each other will be considered at the same position.  在此距离内的任何基因都将被视为处于同一位置。
            clss_to_weight (list, optional): List of labels to weight in the trainer's weighted random sampler. Defaults to [].
            assays_to_drop (list, optional): List of assays to drop from the dataset. Defaults to [].
            do_gene_pos (Union[bool, str], optional): Whether to use gene positions. Defaults to True.
            max_len (int, optional): The maximum length of the input tensor. Defaults to 1000.
            add_zero_genes (int, optional): The number of zero genes to add to the input tensor. Defaults to 100.
            how (str, optional): The method to use for the collator. Defaults to "random expr".
            organism_name (str, optional): The name of the organism. Defaults to "organism_ontology_term_id".
            tp_name (Optional[str], optional): The name of the timepoint. Defaults to None.
            hierarchical_clss (list, optional): List of hierarchical classes. Defaults to [].
            all_clss (list, optional): List of all classes. Defaults to ["organism_ontology_term_id"].
            **kwargs: Additional keyword arguments passed to the pytorch DataLoader.

            see @file data.py and @file collator.py for more details about some of the parameters
        """
        if collection_name is not None:
            mdataset = Dataset(
                ln.Collection.filter(name=collection_name).first(),
                organisms=organisms,
                obs=all_clss,
                hierarchical_clss=hierarchical_clss,
            )
            # print(mdataset)
        # and location
        self.gene_pos = None
        if do_gene_pos:
            if type(do_gene_pos) is str:
                print("seeing a string: loading gene positions as biomart parquet file")
                biomart = pd.read_parquet(do_gene_pos)
            else:
                # and annotations
                if organisms != ["NCBITaxon:9606"]:
                    raise ValueError(
                        "need to provide your own table as this automated function only works for humans for now"
                    )
                biomart = getBiomartTable(
                    attributes=["start_position", "chromosome_name"],
                    useCache=True,
                ).set_index("ensembl_gene_id")
                biomart = biomart.loc[~biomart.index.duplicated(keep="first")]
                biomart = biomart.sort_values(by=["chromosome_name", "start_position"])
                c = []
                i = 0
                prev_position = -100000
                prev_chromosome = None
                for _, r in biomart.iterrows():
                    if (
                        r["chromosome_name"] != prev_chromosome
                        or r["start_position"] - prev_position > gene_position_tolerance
                    ):
                        i += 1
                    c.append(i)
                    prev_position = r["start_position"]
                    prev_chromosome = r["chromosome_name"]
                print(f"reduced the size to {len(set(c))/len(biomart)}")
                biomart["pos"] = c
            mdataset.genedf = mdataset.genedf.join(biomart, how="inner")
            self.gene_pos = mdataset.genedf["pos"].astype(int).tolist()

        if gene_embeddings != "":
            mdataset.genedf = mdataset.genedf.join(
                pd.read_parquet(gene_embeddings), how="inner"
            )
            if do_gene_pos:
                self.gene_pos = mdataset.genedf["pos"].tolist()
        self.classes = {k: len(v) for k, v in mdataset.class_topred.items()}
        # we might want not to order the genes by expression (or do it?)
        # we might want to not introduce zeros and
        if use_default_col:
            kwargs["collate_fn"] = Collator(
                organisms=organisms,
                how=how,
                valid_genes=mdataset.genedf.index.tolist(),
                max_len=max_len,
                add_zero_genes=add_zero_genes,
                org_to_id=mdataset.encoder[organism_name],
                tp_name=tp_name,
                organism_name=organism_name,
                class_names=clss_to_weight,
            )
        self.validation_split = validation_split
        self.test_split = test_split
        self.dataset = mdataset
        self.kwargs = kwargs
        if "sampler" in self.kwargs:
            self.kwargs.pop("sampler")
        self.assays_to_drop = assays_to_drop
        self.n_samples = len(mdataset)
        self.weight_scaler = weight_scaler
        self.train_oversampling_per_epoch = train_oversampling_per_epoch
        self.clss_to_weight = clss_to_weight
        self.train_weights = None
        self.train_labels = None
        self.test_datasets = []
        self.test_idx = []
        super().__init__()

    def __repr__(self):
        return (
            f"DataLoader(\n"
            f"\twith a dataset=({self.dataset.__repr__()}\n)\n"
            f"\tvalidation_split={self.validation_split},\n"
            f"\ttest_split={self.test_split},\n"
            f"\tn_samples={self.n_samples},\n"
            f"\tweight_scaler={self.weight_scaler},\n"
            f"\ttrain_oversampling_per_epoch={self.train_oversampling_per_epoch},\n"
            f"\tassays_to_drop={self.assays_to_drop},\n"
            f"\tnum_datasets={len(self.dataset.mapped_dataset.storages)},\n"
            f"\ttest datasets={str(self.test_datasets)},\n"
            f"perc test: {str(len(self.test_idx) / self.n_samples)},\n"
            f"\tclss_to_weight={self.clss_to_weight}\n"
            + (
                (
                    "\twith train_dataset size of=("
                    + str((self.train_weights != 0).sum())
                    + ")\n)"
                )
                if self.train_weights is not None
                else ")"
            )
        )

    @property
    def decoders(self): #返回解码器，用于将编码后的标签转换回原始标签。
        """
        decoders the decoders for any labels that would have been encoded

        Returns:
            dict[str, dict[int, str]]
        """
        decoders = {}
        for k, v in self.dataset.encoder.items():
            decoders[k] = {va: ke for ke, va in v.items()}
        return decoders

    @property
    def labels_hierarchy(self): #返回标签的层次结构。
        """
        labels_hierarchy the hierarchy of labels for any cls that would have a hierarchy

        Returns:
            dict[str, dict[str, str]]
        """
        labels_hierarchy = {}
        for k, dic in self.dataset.labels_groupings.items():
            rdic = {}
            for sk, v in dic.items():
                rdic[self.dataset.encoder[k][sk]] = [
                    self.dataset.encoder[k][i] for i in list(v)
                ]
            labels_hierarchy[k] = rdic
        return labels_hierarchy

    @property
    def genes(self): #返回数据集中使用的基因列表。
        """
        genes the genes used in this datamodule

        Returns:
            list
        """
        return self.dataset.genedf.index.tolist()

    @property
    def num_datasets(self): #返回数据集的数量。
        return len(self.dataset.mapped_dataset.storages)

    def setup(self, stage=None):
        """
        setup method is used to prepare the data for the training, validation, and test sets.
        It shuffles the data, calculates weights for each set, and creates samplers for each set.
       准备数据集的训练、验证和测试集。它根据 validation_split 和 test_split 参数划分数据集，并计算每个类别的权重。
        Args:
            stage (str, optional): The stage of the model training process.
            It can be either 'fit' or 'test'. Defaults to None.
        """
        if len(self.clss_to_weight) > 0 and self.weight_scaler > 0:
            weights, labels = self.dataset.get_label_weights(
                self.clss_to_weight, scaler=self.weight_scaler
            )
        else:
            weights = np.ones(1)
            labels = np.zeros(self.n_samples)
        if isinstance(self.validation_split, int):
            len_valid = self.validation_split
        else:
            len_valid = int(self.n_samples * self.validation_split)
        if isinstance(self.test_split, int):
            len_test = self.test_split
        else:
            len_test = int(self.n_samples * self.test_split)
        assert (
            len_test + len_valid < self.n_samples
        ), "test set + valid set size is configured to be larger than entire dataset."

        idx_full = []
        if len(self.assays_to_drop) > 0:
            for i, a in enumerate(
                self.dataset.mapped_dataset.get_merged_labels("assay_ontology_term_id")
            ):
                if a not in self.assays_to_drop:
                    idx_full.append(i)
            idx_full = np.array(idx_full)
        else:
            idx_full = np.arange(self.n_samples)
        if len_test > 0:
            # this way we work on some never seen datasets
            # keeping at least one
            len_test = (
                len_test
                if len_test > self.dataset.mapped_dataset.n_obs_list[0]
                else self.dataset.mapped_dataset.n_obs_list[0]
            )
            cs = 0
            for i, c in enumerate(self.dataset.mapped_dataset.n_obs_list):
                if cs + c > len_test:
                    break
                else:
                    self.test_datasets.append(
                        self.dataset.mapped_dataset.path_list[i].path
                    )
                    cs += c
            len_test = cs
            self.test_idx = idx_full[:len_test]
            idx_full = idx_full[len_test:]
        else:
            self.test_idx = None

        np.random.shuffle(idx_full)
        if len_valid > 0:
            self.valid_idx = idx_full[:len_valid].copy()
            idx_full = idx_full[len_valid:]
        else:
            self.valid_idx = None
        weights = np.concatenate([weights, np.zeros(1)])
        labels[~np.isin(np.arange(self.n_samples), idx_full)] = len(weights) - 1

        self.train_weights = weights
        self.train_labels = labels
        self.idx_full = idx_full
        return self.test_datasets

    def train_dataloader(self, **kwargs): #返回训练集的数据加载器，使用 LabelWeightedSampler 进行加权随机采样。
        # train_sampler = WeightedRandomSampler(
        #    self.train_weights[self.train_labels],
        #    int(self.n_samples*self.train_oversampling_per_epoch),
        #    replacement=True,
        # )
        train_sampler = LabelWeightedSampler(
            self.train_weights,
            self.train_labels,
            num_samples=int(self.n_samples * self.train_oversampling_per_epoch),
        )
        return DataLoader(self.dataset, sampler=train_sampler, **self.kwargs, **kwargs)

    def val_dataloader(self): #返回验证集的数据加载器，使用 SubsetRandomSampler 进行随机采样。
        return (
            DataLoader(
                self.dataset, sampler=SubsetRandomSampler(self.valid_idx), **self.kwargs
            )
            if self.valid_idx is not None
            else None
        )

    def test_dataloader(self): # 返回测试集的数据加载器，使用 SequentialSampler 进行顺序采样。
        return (
            DataLoader(
                self.dataset, sampler=SequentialSampler(self.test_idx), **self.kwargs
            )
            if self.test_idx is not None
            else None
        )

    def predict_dataloader(self): #返回用于预测的数据加载器，使用 SubsetRandomSampler 进行随机采样。
        return DataLoader(
            self.dataset, sampler=SubsetRandomSampler(self.idx_full), **self.kwargs
        )

    # def teardown(self):
    # clean up state after the trainer stops, delete files...
    # called on every process in DDP
    # pass


class LabelWeightedSampler(Sampler[int]): #LabelWeightedSampler 是一个自定义的采样器，用于在训练过程中对类别进行加权采样。
    label_weights: Sequence[float]
    klass_indices: Sequence[Sequence[int]]
    num_samples: int
    # 它根据类别的权重和标签生成样本索引，确保在训练过程中类别分布均衡。
    # when we use, just set weights for each classes(here is: np.ones(num_classes)), and labels of a dataset. 当我们使用时，只需为每个类别（此处为：np.ones(num_classes)）和数据集的标签设置权重。
    # this will result a class-balanced sampling, no matter how imbalance the labels are. 无论标签有多不平衡，这都将导致类平衡采样。
    # NOTE: here we use replacement=True, you can change it if you don't upsample a class. 这里我们使用 replacement=True，如果不对某个类进行上采样，也可以更改。
    def __init__( #构造函数
        self, label_weights: Sequence[float], labels: Sequence[int], num_samples: int
    ) -> None:
        """
        接受 label_weights（类别权重）、labels（数据集标签）和 num_samples（样本数量）作为参数。计算每个类别的权重，并生成类别的索引列表。
        :param label_weights: list(len=num_classes)[float], weights for each class.
        :param labels: list(len=dataset_len)[int], labels of a dataset.
        :param num_samples: number of samples.
        """

        super(LabelWeightedSampler, self).__init__(None)
        # reweight labels from counter otherwsie same weight to labels that have many elements vs a few
        label_weights = np.array(label_weights) * np.bincount(labels)

        self.label_weights = torch.as_tensor(label_weights, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.int)
        self.num_samples = num_samples
        # list of tensor.
        self.klass_indices = [
            (self.labels == i_klass).nonzero().squeeze(1)
            for i_klass in range(len(label_weights))
        ]

    def __iter__(self): #生成加权采样的样本索引。
        sample_labels = torch.multinomial(
            self.label_weights, num_samples=self.num_samples, replacement=True
        )
        sample_indices = torch.empty_like(sample_labels)
        for i_klass, klass_index in enumerate(self.klass_indices):
            if klass_index.numel() == 0:
                continue
            left_inds = (sample_labels == i_klass).nonzero().squeeze(1)
            right_inds = torch.randint(len(klass_index), size=(len(left_inds),))
            sample_indices[left_inds] = klass_index[right_inds]
        yield from iter(sample_indices.tolist())

    def __len__(self): #返回样本数量。
        return self.num_samples

"""
这个文件的主要功能是管理数据集的加载和处理，特别是针对基因表达数据。它支持加权采样、基因位置信息的处理、以及数据集的划分。
DataModule 类可以与 PyTorch Lightning 框架无缝集成，简化了数据加载和训练过程。
LabelWeightedSampler 类则提供了类别均衡的采样功能，适用于类别不平衡的数据集。
"""
