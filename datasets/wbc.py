import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden


@DATASET_REGISTRY.register()
class WBC(DatasetBase):
   

    dataset_dir = "WBC"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "WBC")

        self.split_path = os.path.join(self.dataset_dir, "split_WBC.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_and_split_data(self.image_dir)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        # Few-shot
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Base/New subsampling
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    def generate_fewshot_dataset_(self,num_shots, split):
        '''
            # 这是一个辅助函数，用来调用（继承来的）generate_fewshot_dataset 方法。
            # 它根据传入的 split 字符串（"train" 或 "val"）来决定
            # 是对 self.train_x 还是 self.val 进行 few-shot 采样。
            # 名字里的下划线 _ 通常表示这是一个内部使用的方法。
        '''

        print('num_shots is ',num_shots)

        if split == "train":
            few_shot_data = self.generate_fewshot_dataset(self.train_x, num_shots=num_shots)
        elif split == "val":
            few_shot_data = self.generate_fewshot_dataset(self.val, num_shots=num_shots)
    
        return few_shot_data
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, item.label, item.classname))
            return out

        split = {
            "train": _extract(train),
            "val": _extract(val),
            "test": _extract(test),
        }
        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                out.append(Datum(impath=impath, label=int(label), classname=classname))
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])
        return train, val, test

    @staticmethod
    def read_and_split_data(image_dir, p_trn=0.7, p_val=0.1, ignored=None, new_cnames=None):
        """
        Folder-per-class split:
            image_dir/<class_name>/*.*
        """
        ignored = ignored or []

        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            return [Datum(impath=im, label=y, classname=c) for im in ims]

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]

            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0, f"Not enough images in {category_dir}"

            cname = new_cnames.get(category, category) if new_cnames else category

            train.extend(_collate(images[:n_train], label, cname))
            val.extend(_collate(images[n_train:n_train + n_val], label, cname))
            test.extend(_collate(images[n_train + n_val:], label, cname))

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        assert subsample in ["all", "base", "new"]
        if subsample == "all":
            return args

        dataset = args[0]
        labels = sorted({item.label for item in dataset})
        n = len(labels)
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        selected = labels[:m] if subsample == "base" else labels[m:]
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dset in args:
            dnew = []
            for item in dset:
                if item.label not in selected:
                    continue
                dnew.append(Datum(impath=item.impath, label=relabeler[item.label], classname=item.classname))
            output.append(dnew)

        return output
