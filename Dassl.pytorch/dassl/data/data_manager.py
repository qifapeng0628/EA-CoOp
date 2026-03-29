


# medgazecoop data manager


import os
import hashlib
import random
import numpy as np

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import functional as TF

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import build_transform

INTERPOLATION_MODES = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


def _is_autoaugment_policy(op) -> bool:
    return op.__class__.__name__ in {"ImageNetPolicy", "CIFAR10Policy", "SVHNPolicy"}


def _is_randaugment(op) -> bool:
    return op.__class__.__name__ in {"RandAugment", "RandAugment2", "RandAugmentFixMatch"}


def _is_random2d_translation(op) -> bool:
    return op.__class__.__name__ == "Random2DTranslation" and hasattr(op, "height") and hasattr(op, "width")


def _apply_random2d_translation_pair(op, img: Image.Image, mask: Image.Image):
    """
    Paired version of transforms.Random2DTranslation (dassl.data.transforms).
    Mirrors its randomness exactly, but uses NEAREST for mask interpolation.
    """
    if random.uniform(0, 1) > op.p:
        img_out = img.resize((op.width, op.height), op.interpolation)
        mask_out = mask.resize((op.width, op.height), Image.NEAREST)
        return img_out, mask_out

    new_width = int(round(op.width * 1.125))
    new_height = int(round(op.height * 1.125))
    resized_img = img.resize((new_width, new_height), op.interpolation)
    resized_mask = mask.resize((new_width, new_height), Image.NEAREST)

    x_maxrange = new_width - op.width
    y_maxrange = new_height - op.height
    x1 = int(round(random.uniform(0, x_maxrange)))
    y1 = int(round(random.uniform(0, y_maxrange)))

    img_out = resized_img.crop((x1, y1, x1 + op.width, y1 + op.height))
    mask_out = resized_mask.crop((x1, y1, x1 + op.width, y1 + op.height))
    return img_out, mask_out


def _apply_randaugment_mask_op(mask: Image.Image, op_name: str, v: float) -> Image.Image:
    """
    Apply ONLY spatial RandAugment operations to the mask, using NEAREST + fill=0.

    IMPORTANT:
    - randaugment.py uses internal random() draws for sign flips in Shear/Translate/Rotate.
      The caller must restore the same random state before calling this function to keep
      the mask transform identical to the image transform.
    """
    import PIL
    import PIL.ImageOps

    if op_name == "ShearX":
        if random.random() > 0.5:
            v = -v
        return mask.transform(mask.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0),
                              resample=Image.NEAREST, fillcolor=0)

    if op_name == "ShearY":
        if random.random() > 0.5:
            v = -v
        return mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0),
                              resample=Image.NEAREST, fillcolor=0)

    if op_name == "TranslateX":
        if random.random() > 0.5:
            v = -v
        v_px = v * mask.size[0]
        return mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, v_px, 0, 1, 0),
                              resample=Image.NEAREST, fillcolor=0)

    if op_name == "TranslateY":
        if random.random() > 0.5:
            v = -v
        v_px = v * mask.size[1]
        return mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v_px),
                              resample=Image.NEAREST, fillcolor=0)

    if op_name == "TranslateXabs":
        if random.random() > 0.5:
            v = -v
        return mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0),
                              resample=Image.NEAREST, fillcolor=0)

    if op_name == "TranslateYabs":
        if random.random() > 0.5:
            v = -v
        return mask.transform(mask.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v),
                              resample=Image.NEAREST, fillcolor=0)

    if op_name == "Rotate":
        if random.random() > 0.5:
            v = -v
        return mask.rotate(v, resample=Image.NEAREST, fillcolor=0)

    if op_name == "Flip":
        return PIL.ImageOps.mirror(mask)

    # Non-spatial ops -> keep mask unchanged
    return mask


def _apply_randaugment_pair(op, img: Image.Image, mask: Image.Image):
    """
    Paired application for RandAugment / RandAugment2 / RandAugmentFixMatch.
    Same sampled ops for image+mask; mask only receives spatial ops.
    """
    augment_list = getattr(op, "augment_list", None)
    if augment_list is None:
        return img, mask

    n = getattr(op, "n", 2)
    cls = op.__class__.__name__

    ops = random.choices(augment_list, k=n)

    for opfunc, minval, maxval in ops:
        name = getattr(opfunc, "__name__", "")

        if cls == "RandAugment":
            m = getattr(op, "m", 10)
            val = (m / 30.0) * (maxval - minval) + minval

            st = random.getstate()
            img = opfunc(img, val)
            random.setstate(st)
            mask = _apply_randaugment_mask_op(mask, name, val)
            continue

        if cls == "RandAugment2":
            p = getattr(op, "p", 0.6)
            if random.random() > p:
                continue
            m = random.random()
            val = m * (maxval - minval) + minval

            st = random.getstate()
            img = opfunc(img, val)
            random.setstate(st)
            mask = _apply_randaugment_mask_op(mask, name, val)
            continue

        if cls == "RandAugmentFixMatch":
            m = random.random()
            val = m * (maxval - minval) + minval

            st = random.getstate()
            img = opfunc(img, val)
            random.setstate(st)
            mask = _apply_randaugment_mask_op(mask, name, val)
            continue

    return img, mask


def _apply_pair_transform(
    tfm,
    img: Image.Image,
    mask: Image.Image,
    skip_autoaugment: bool = False,
    return_pil: bool = False,
):
    """
    Apply transform to BOTH image and mask with aligned spatial transforms.

    - Spatial transforms are applied to image & mask with the SAME params.
    - Non-spatial transforms (ColorJitter, Blur, etc.) apply to image only.
    - Mask always uses NEAREST interpolation for spatial ops.
    - RandAugment: applied to image normally; spatial ops (Shear/Translate/Rotate)
      are replicated on mask with the same random state.
    - If return_pil=True, also return aligned PIL image/mask right before ToTensor.

    NOTE: skip_autoaugment defaults to False now — RandAugment IS applied
    with paired spatial handling. Set True only if you want to skip it entirely.
    """

    # Ensure modes
    if img.mode != "RGB":
        img = img.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")

    # Expand ops
    if tfm is None:
        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        if return_pil:
            return img_t, mask_t, img, mask
        return img_t, mask_t

    ops = tfm.transforms if isinstance(tfm, T.Compose) else [tfm]

    img_obj = img
    mask_obj = mask
    img_t = None
    mask_t = None

    # Debug PIL snapshots (aligned)
    dbg_img_pil = None
    dbg_mask_pil = None

    for op in ops:
        # ---------------------------
        # Paired spatial transforms
        # ---------------------------
        if isinstance(op, T.RandomHorizontalFlip):
            if random.random() < op.p:
                img_obj = TF.hflip(img_obj)
                mask_obj = TF.hflip(mask_obj)
            continue

        if isinstance(op, T.CenterCrop):
            img_obj = TF.center_crop(img_obj, op.size)
            mask_obj = TF.center_crop(mask_obj, op.size)
            continue

        if isinstance(op, T.RandomResizedCrop):
            i, j, h, w = op.get_params(img_obj, op.scale, op.ratio)
            img_obj = TF.resized_crop(
                img_obj, i, j, h, w, op.size, interpolation=op.interpolation
            )
            mask_obj = TF.resized_crop(
                mask_obj, i, j, h, w, op.size, interpolation=TF.InterpolationMode.NEAREST
            )
            continue

        if isinstance(op, T.Resize):
            img_obj = TF.resize(
                img_obj, op.size, interpolation=op.interpolation, antialias=getattr(op, "antialias", None)
            )
            mask_obj = TF.resize(
                mask_obj, op.size, interpolation=TF.InterpolationMode.NEAREST, antialias=None
            )
            continue

        if isinstance(op, T.RandomCrop):
            if op.padding is not None:
                img_obj = TF.pad(img_obj, op.padding, fill=op.fill, padding_mode=op.padding_mode)
                mask_obj = TF.pad(mask_obj, op.padding, fill=0, padding_mode=op.padding_mode)

            i, j, h, w = T.RandomCrop.get_params(img_obj, output_size=op.size)
            img_obj = TF.crop(img_obj, i, j, h, w)
            mask_obj = TF.crop(mask_obj, i, j, h, w)
            continue

        # ---------------------------
        # RandAugment — now applied with paired spatial ops
        # ---------------------------
        if _is_randaugment(op):
            if skip_autoaugment:
                continue
            # Apply RandAugment to image; replicate spatial ops on mask
            img_obj, mask_obj = _apply_randaugment_pair(op, img_obj, mask_obj)
            continue

        if _is_random2d_translation(op):
            try:
                img_obj, mask_obj = _apply_random2d_translation_pair(op, img_obj, mask_obj)
            except Exception:
                try:
                    img_obj = op(img_obj)
                except Exception:
                    pass
            continue

        # ---------------------------
        # Tensor conversion / normalize
        # ---------------------------
        if isinstance(op, T.ToTensor):
            # Capture aligned PIL before tensorization
            if isinstance(img_obj, Image.Image) and isinstance(mask_obj, Image.Image):
                dbg_img_pil = img_obj
                dbg_mask_pil = mask_obj

            img_t = op(img_obj)              # [C,H,W] float
            mask_t = TF.to_tensor(mask_obj)  # [1,H,W] float in [0,1]

            # Optional binarization (OFF by default)
            if os.getenv("GAZE_BINARIZE", "0") == "1":
                thr = float(os.getenv("GAZE_MASK_THR", "0.03"))
                mask_t = (mask_t >= thr).to(dtype=mask_t.dtype)

            continue

        if isinstance(op, T.Normalize):
            if img_t is None:
                img_t = TF.to_tensor(img_obj)
            img_t = op(img_t)
            continue

        # ---------------------------
        # Other transforms (image only)
        # ---------------------------
        try:
            img_obj = op(img_obj)
        except Exception:
            if img_t is None:
                img_t = TF.to_tensor(img_obj)
            img_t = op(img_t)

    # ---------------------------
    # Finalize tensors
    # ---------------------------
    if img_t is None:
        img_t = TF.to_tensor(img_obj)
    if mask_t is None:
        mask_t = TF.to_tensor(mask_obj)

    # ---------------------------
    # Optional: return PIL debug snapshots
    # ---------------------------
    if return_pil:
        if dbg_img_pil is None and isinstance(img_obj, Image.Image):
            dbg_img_pil = img_obj
        if dbg_mask_pil is None and isinstance(mask_obj, Image.Image):
            dbg_mask_pil = mask_obj
        return img_t, mask_t, dbg_img_pil, dbg_mask_pil

    return img_t, mask_t


# ======================================================================
# Threshold-area adaptive crop
# ======================================================================

def _compute_gaze_crop_area(
    mask_np,
    img_tensor,
    crop_k=1.5,
    smin=32,
    smax=224,
    area_thr=0.6,
    adaptive=True,
    jitter=0,
    target_size=224,
    eps=1e-6,
):
    
    import numpy as np
    import torch

    H, W = mask_np.shape

    
    m_mean = float(mask_np.mean())
    m_sum = float(mask_np.sum())
    if (m_sum <= eps) or (m_mean <= eps) or (m_mean >= 1.0 - eps):
        valid = False
        s = int(max(8, min(smax, H, W)))
        cy, cx = H // 2, W // 2
        y0 = max(0, min(cy - s // 2, H - s))
        x0 = max(0, min(cx - s // 2, W - s))
        patch = img_tensor[:, y0:y0 + s, x0:x0 + s].unsqueeze(0)
        crop = torch.nn.functional.interpolate(
            patch, size=(target_size, target_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        return crop, valid

    valid = True

    
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    wy = mask_np.sum(axis=1)  # [H]
    wx = mask_np.sum(axis=0)  # [W]
    denom = float(mask_np.sum()) + eps
    cy = float((wy * ys).sum() / denom)
    cx = float((wx * xs).sum() / denom)

    # Crop size
    if adaptive:
        ar = float((mask_np >= float(area_thr)).mean())
        deb = float(mask_np.mean())
        # ar = max(ar, float(mask_np.mean()))  # fallback for soft heatmaps
        
        ar = float(np.clip(ar, 0.0, 1.0))
        r = float(np.sqrt(ar))
        base = float(min(H, W))
        crop_size = int(round(base * r * float(crop_k)))
    else:
        crop_size = int(smax)

    crop_size = int(np.clip(crop_size, int(smin), int(smax)))
    crop_size = int(max(8, min(crop_size, H, W)))

    # Optional center jitter (train only)
    if jitter > 0:
        cy += float(np.random.uniform(-jitter, jitter))
        cx += float(np.random.uniform(-jitter, jitter))

    half = crop_size / 2.0
    y0 = int(round(cy - half))
    x0 = int(round(cx - half))
    y0 = max(0, min(y0, H - crop_size))
    x0 = max(0, min(x0, W - crop_size))

    patch = img_tensor[:, y0:y0 + crop_size, x0:x0 + crop_size].unsqueeze(0)
    crop = torch.nn.functional.interpolate(
        patch, size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze(0)

    return crop, valid


def build_data_loader(
        cfg,
        sampler_type="SequentialSampler",
        data_source=None,
        batch_size=64,
        n_domain=0,
        n_ins=2,
        tfm=None,
        is_train=True,
        dataset_wrapper=None,
):
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins,
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=False,
    )
    assert len(data_loader) > 0
    return data_loader


class DataManager:

    def __init__(
            self,
            cfg,
            custom_tfm_train=None,
            custom_tfm_test=None,
            dataset_wrapper=None
    ):
        dataset = build_dataset(cfg)

        # ---- DEBUG: train_x signature (optional) ----
        if os.environ.get("DATA_DEBUG", "0") == "1":
            try:
                impaths = [d.impath for d in dataset.train_x]
                sig = hashlib.md5(("\n".join(sorted(impaths))).encode("utf-8")).hexdigest()
                print("\n[DATA_DEBUG train_x] num:", len(impaths), "md5:", sig)
                print("[DATA_DEBUG train_x] first 20 impaths:")
                for p in impaths[:20]:
                    print(" ", p)
                print()
            except Exception as e:
                print("[DATA_DEBUG train_x] failed:", repr(e))
        # ---- END DEBUG ----


        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper,
        )

        # ---- DEBUG: first batch structure (optional) ----
        if os.environ.get("DATA_DEBUG_PEEK_BATCH", "0") == "1":
            try:
                import random as _py_random
                import numpy as _np
                _st_py = _py_random.getstate()
                _st_np = _np.random.get_state()
                _st_th = torch.get_rng_state()
                _st_cuda = None
                if torch.cuda.is_available():
                    _st_cuda = torch.cuda.get_rng_state_all()
        
                if getattr(cfg.DATALOADER, "NUM_WORKERS", 0) != 0:
                    print("[DATA_DEBUG FIRST BATCH] Warning: NUM_WORKERS != 0; batch peek may perturb worker RNG.")
        
                _batch = next(iter(train_loader_x))
                print("\n[DATA_DEBUG FIRST BATCH]")
                try:
                    print("keys:", list(_batch.keys()))
                except Exception:
                    pass
        
                _img = _batch.get("img", None)
                print("type(batch['img']):", type(_img))
                if isinstance(_img, (list, tuple)):
                    for i, v in enumerate(_img):
                        try:
                            print(f"  view[{i}] shape:", tuple(v.shape), "dtype:", v.dtype)
                        except Exception:
                            print(f"  view[{i}] type:", type(v))
                else:
                    try:
                        print("  img shape:", tuple(_img.shape), "dtype:", _img.dtype)
                    except Exception:
                        print("  img is None or non-tensor")
        
                _has_mask = ("mask" in _batch)
                print("has 'mask' key:", _has_mask)
                if _has_mask and _batch["mask"] is not None:
                    _m = _batch["mask"]
                    try:
                        print("  mask shape:", tuple(_m.shape), "dtype:", _m.dtype,
                              "min/max/sum:", float(_m.min()), float(_m.max()), float(_m.sum()))
                    except Exception:
                        print("  mask type:", type(_m))

                _has_crop = ("crop_img" in _batch)
                print("has 'crop_img' key:", _has_crop)
                if _has_crop and _batch["crop_img"] is not None:
                    _c = _batch["crop_img"]
                    try:
                        print("  crop_img shape:", tuple(_c.shape), "dtype:", _c.dtype)
                    except Exception:
                        print("  crop_img type:", type(_c))

                print()
        
                # restore RNG
                _py_random.setstate(_st_py)
                _np.random.set_state(_st_np)
                torch.set_rng_state(_st_th)
                if _st_cuda is not None:
                    torch.cuda.set_rng_state_all(_st_cuda)
            except Exception as e:
                print("[DATA_DEBUG FIRST BATCH] failed:", repr(e))
        # ---- END DEBUG ----


        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper,
            )

        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper,
            )

        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper,
        )

        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        print("***** Dataset statistics *****")
        print("  Dataset: {}".format(cfg.DATASET.NAME))
        if cfg.DATASET.SOURCE_DOMAINS:
            print("  Source domains: {}".format(cfg.DATASET.SOURCE_DOMAINS))
        if cfg.DATASET.TARGET_DOMAINS:
            print("  Target domains: {}".format(cfg.DATASET.TARGET_DOMAINS))
        print("  # classes: {:,}".format(self.num_classes))
        print("  # train_x: {:,}".format(len(self.dataset.train_x)))
        if self.dataset.train_u:
            print("  # train_u: {:,}".format(len(self.dataset.train_u)))
        if self.dataset.val:
            print("  # val: {:,}".format(len(self.dataset.val)))
        print("  # test: {:,}".format(len(self.dataset.test)))


class DatasetWrapper(TorchDataset):
    

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train

        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        self.use_gaze = (os.environ.get("USE_GAZE_MASK", "0") == "1") and is_train
        data_root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        default_root = os.path.join(data_root, "select", str(cfg.DATASET.NAME))
        self.gaze_root = os.environ.get("GAZE_HEATMAP_ROOT", default_root)

        
        self.gaze_crop_k = float(os.environ.get("GAZE_CROP_K", "1.0"))
        self.gaze_crop_smin = int(os.environ.get("GAZE_CROP_SMIN", "32"))
        self.gaze_crop_smax = int(os.environ.get("GAZE_CROP_SMAX",
                                   os.environ.get("GAZE_CROP_SIZE", "224")))
        self.gaze_crop_jitter = int(os.environ.get("GAZE_CROP_JITTER", "0"))
        self.gaze_area_thr = float(os.environ.get("GAZE_AREA_THR", "0.35"))
        self.gaze_adaptive_crop = (os.environ.get("GAZE_ADAPTIVE_CROP", "1") == "1")
        self.gaze_target_size = int(cfg.INPUT.SIZE[0]) if hasattr(cfg.INPUT, 'SIZE') else 224

        
        self.gaze_debug = (os.environ.get("GAZE_DEBUG", "0") == "1") and self.use_gaze
        self.gaze_debug_max = int(os.environ.get("GAZE_DEBUG_MAX", "20"))
        self._gaze_dbg_cnt = 0

        
        self.gaze_sanity = (os.environ.get("GAZE_SANITY_SAVE", "0") == "1") and self.use_gaze
        self.gaze_sanity_dir = os.environ.get("GAZE_SANITY_DIR", "./gaze_sanity")
        self.gaze_sanity_max = int(os.environ.get("GAZE_SANITY_MAX", "16"))
        self._gaze_sanity_cnt = 0
        if self.gaze_sanity:
            os.makedirs(self.gaze_sanity_dir, exist_ok=True)

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times because transform is None".format(self.k_tfm)
            )

        
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def _resolve_heatmap_path(self, item):
        stem = os.path.splitext(os.path.basename(item.impath))[0]
        cls_from_item = getattr(item, "classname", None)
        cls_from_path = os.path.basename(os.path.dirname(item.impath))

        cls_candidates = []
        if cls_from_item:
            cls_candidates.append(str(cls_from_item))
        if cls_from_path and cls_from_path not in cls_candidates:
            cls_candidates.append(str(cls_from_path))

        for cls in cls_candidates:
            p = os.path.join(self.gaze_root, cls, f"{stem}_heatmap.png")
            if os.path.exists(p):
                return p

        return None

    def _read_heatmap(self, item, img_size):
        hp = self._resolve_heatmap_path(item)

        info = {
            "heatmap_path": hp,
            "exists": False,
            "raw_min": None,
            "raw_max": None,
            "raw_sum": None,
            "resized_to_img": False,
            "exception": None,
            "img_size": img_size,
        }

        if hp is None:
            return Image.new("L", img_size, color=0), info

        try:
            hm = Image.open(hp).convert("L")
            info["exists"] = True

            arr = np.array(hm)
            info["raw_min"] = int(arr.min()) if arr.size > 0 else 0
            info["raw_max"] = int(arr.max()) if arr.size > 0 else 0
            info["raw_sum"] = float(arr.sum())

        except Exception as e:
            info["exception"] = repr(e)
            return Image.new("L", img_size, color=0), info

        if hm.size != img_size:
            hm = hm.resize(img_size, resample=Image.NEAREST)
            info["resized_to_img"] = True

        return hm, info

    def _compute_crop(self, img_tensor, mask_tensor):
        """
        Compute gaze-guided crop using threshold-based area ratio.

        Args:
            img_tensor:  [C, H, W] augmented image tensor
            mask_tensor: [1, H, W] aligned mask tensor in [0,1]

        Returns:
            crop_img:   [C, target_size, target_size] tensor
            crop_valid: bool
        """
        mask_np = mask_tensor.squeeze(0).numpy().astype(np.float32)

        
        mmax = mask_np.max()
        if mmax > 1.5:
            mask_np = mask_np / (mmax + 1e-6)
        mask_np = np.clip(mask_np, 0.0, 1.0)

        H, W = mask_np.shape
        smax = int(max(8, min(self.gaze_crop_smax, W, H)))
        smin = int(max(8, min(self.gaze_crop_smin, W, H)))
        smin = min(smin, smax)

        crop_img, valid = _compute_gaze_crop_area(
            mask_np,
            img_tensor,
            crop_k=self.gaze_crop_k,
            smin=smin,
            smax=smax,
            area_thr=self.gaze_area_thr,
            adaptive=self.gaze_adaptive_crop,
            jitter=self.gaze_crop_jitter if self.is_train else 0,
            target_size=self.gaze_target_size,
        )
        return crop_img, valid


    def _save_gaze_sanity(self, img_pil: Image.Image, mask_pil: Image.Image, item, aug_i: int = 0):
        """Save debug overlay and crop preview using the SAME area-based crop sizing."""
        if not getattr(self, "gaze_sanity", False):
            return

        try:
            if getattr(self, "_gaze_sanity_cnt", 0) >= int(getattr(self, "gaze_sanity_max", 0)):
                return

            os.makedirs(getattr(self, "gaze_sanity_dir", "./gaze_sanity"), exist_ok=True)

            img_rgb = img_pil.convert("RGB")
            if mask_pil.size != img_rgb.size:
                mask_pil = mask_pil.resize(img_rgb.size, resample=Image.NEAREST)

            m = np.asarray(mask_pil.convert("L"), dtype=np.float32)
            if m.max() > 1.0:
                m = m / (m.max() + 1e-6)
            m = np.clip(m, 0.0, 1.0)

            H, W = m.shape
            eps = 1e-6
            mass = float(m.sum())

            # Center-of-mass (soft)
            if mass <= eps:
                cy, cx = H / 2.0, W / 2.0
            else:
                ys = np.arange(H, dtype=np.float32)
                xs = np.arange(W, dtype=np.float32)
                wy = m.sum(axis=1)  # [H]
                wx = m.sum(axis=0)  # [W]
                cy = float((wy * ys).sum() / (mass + eps))
                cx = float((wx * xs).sum() / (mass + eps))

            # Crop size (area ratio; same as _compute_gaze_crop_area)
            smax = int(max(8, min(self.gaze_crop_smax, W, H)))
            smin = int(max(8, min(self.gaze_crop_smin, W, H)))
            smin = min(smin, smax)

            if (mass <= eps) or (not getattr(self, "gaze_adaptive_crop", True)):
                crop_size = smax
            else:
                ar = float((m >= float(getattr(self, "gaze_area_thr", 0.6))).mean())
                # ar = max(ar, float(m.mean()))
                ar = float(np.clip(ar, 0.0, 1.0))
                r = float(np.sqrt(ar))
                base = float(min(H, W))
                crop_size = int(np.clip(base * r * float(self.gaze_crop_k), smin, smax))

            crop_size = int(max(8, min(crop_size, W, H)))

            half = crop_size / 2.0
            left = int(round(cx - half))
            top = int(round(cy - half))
            left = max(0, min(left, W - crop_size))
            top = max(0, min(top, H - crop_size))
            right = left + crop_size
            bottom = top + crop_size

            
            alpha = (m * 180.0).astype(np.uint8)
            heat_rgba = Image.new("RGBA", (W, H), (255, 0, 0, 0))
            heat_rgba.putalpha(Image.fromarray(alpha, mode="L"))

            over_rgba = Image.alpha_composite(img_rgb.convert("RGBA"), heat_rgba)
            over_img = over_rgba.convert("RGB")

            draw = ImageDraw.Draw(over_img)
            draw.rectangle([left, top, right - 1, bottom - 1], outline=(0, 255, 0), width=2)

            cx_i = int(round(cx))
            cy_i = int(round(cy))
            r0 = 3
            draw.ellipse([cx_i - r0, cy_i - r0, cx_i + r0, cy_i + r0], outline=(0, 255, 0), width=2)

            crop224 = img_rgb.crop((left, top, right, bottom)).resize((224, 224), resample=Image.BILINEAR)

            imp = getattr(item, "impath", "img")
            base_name = os.path.splitext(os.path.basename(imp))[0]
            clsname = getattr(item, "classname", os.path.basename(os.path.dirname(imp)))

            wid = 0
            try:
                wi = torch.utils.data.get_worker_info()
                wid = wi.id if wi is not None else 0
            except Exception:
                wid = 0
            pid = os.getpid()

            idx = getattr(self, "_gaze_sanity_cnt", 0)
            tag = f"{idx:04d}_w{wid}_p{pid}_aug{aug_i}_{clsname}_{base_name}"

            over_path = os.path.join(getattr(self, "gaze_sanity_dir", "./gaze_sanity"), tag + "_overlay.png")
            crop_path = os.path.join(getattr(self, "gaze_sanity_dir", "./gaze_sanity"), tag + "_crop224.png")

            over_img.save(over_path)
            crop224.save(crop_path)

            self._gaze_sanity_cnt = idx + 1

        except Exception as e:
            if os.environ.get("GAZE_DEBUG", "0") == "1":
                print(f"[GAZE_SANITY] save failed: {repr(e)}")
            return

    def __getitem__(self, idx):
        item = self.data_source[idx]
        output = {"label": item.label, "domain": item.domain, "impath": item.impath}

        img0 = read_image(item.impath)  # PIL
        if img0.mode != "RGB":
            img0 = img0.convert("RGB")

        mask0 = None
        hm_info = None

        if self.use_gaze:
            mask0, hm_info = self._read_heatmap(item, img0.size)
            if (hm_info["heatmap_path"] is None) or (not hm_info["exists"]):
                raise FileNotFoundError(f"[GAZE] heatmap not found for impath={item.impath} | gaze_root={self.gaze_root}")
            # ---- DEBUG ----
            if self.gaze_debug and (self._gaze_dbg_cnt < self.gaze_debug_max):
                self._gaze_dbg_cnt += 1
                cls_from_item = getattr(item, "classname", None)
                cls_from_path = os.path.basename(os.path.dirname(item.impath))
                stem = os.path.splitext(os.path.basename(item.impath))[0]

                print("\n[GAZE_DEBUG] ------------------------------")
                print("[GAZE_DEBUG] idx:", idx)
                print("[GAZE_DEBUG] impath:", item.impath)
                print("[GAZE_DEBUG] classname(item/path):", cls_from_item, "/", cls_from_path)
                print("[GAZE_DEBUG] stem:", stem)
                print("[GAZE_DEBUG] gaze_root:", self.gaze_root)
                print("[GAZE_DEBUG] heatmap_path:", hm_info["heatmap_path"])
                print("[GAZE_DEBUG] heatmap_exists:", hm_info["exists"])
                if hm_info["exception"] is not None:
                    print("[GAZE_DEBUG] heatmap_open_exception:", hm_info["exception"])
                else:
                    print("[GAZE_DEBUG] raw_heatmap min/max/sum:",
                          hm_info["raw_min"], hm_info["raw_max"], hm_info["raw_sum"])
                    print("[GAZE_DEBUG] resized_to_img:", hm_info["resized_to_img"],
                          "img_size:", hm_info["img_size"],
                          "hm_size:", mask0.size)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    keyname = "img" if (i + 1) == 1 else f"img{i + 1}"
                    img, mask = self._transform_image(tfm, img0, mask0, item=item, debug_tag=keyname)
                    output[keyname] = img
                    if mask is not None:
                        mask_key = "mask" if (i + 1) == 1 else f"mask{i + 1}"
                        output[mask_key] = mask
            else:
                img, mask = self._transform_image(self.transform, img0, mask0, item=item, debug_tag="img")
                output["img"] = img
                if mask is not None:
                    output["mask"] = mask

        
        if self.use_gaze and output.get("mask", None) is not None:
            img_tensor = output["img"]
            mask_tensor = output["mask"]

            # Handle list case (K_TRANSFORMS > 1): use first augmentation
            if isinstance(img_tensor, (list, tuple)):
                img_for_crop = img_tensor[0]
                mask_for_crop = mask_tensor[0] if isinstance(mask_tensor, (list, tuple)) else mask_tensor
            else:
                img_for_crop = img_tensor
                mask_for_crop = mask_tensor

            crop_img, crop_valid = self._compute_crop(img_for_crop, mask_for_crop)
            # if not crop_valid:
            #     print(f"[CROP_INVALID] {item.impath}", flush=True)
            #     print(f"[CROP_INVALID] impath={item.impath} heatmap={hm_info.get('heatmap_path', None) if hm_info else None}", flush=True)
            output["crop_img"] = crop_img
            output["crop_valid"] = torch.tensor(crop_valid, dtype=torch.bool)
        else:
            # No gaze: provide dummy crop_valid=False so collation doesn't fail
            if self.use_gaze:
                C = 3
                H = W = self.gaze_target_size
                output["crop_img"] = torch.zeros(C, H, W)
                output["crop_valid"] = torch.tensor(False, dtype=torch.bool)

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)

        return output

    def _transform_image(self, tfm, img0, mask0, item=None, debug_tag="img"):
        """
        Mirrors original behavior: may augment K times; returns a single tensor if K=1,
        else returns list of tensors.

        If mask0 is provided (training + USE_GAZE_MASK=1), we generate the aligned masks
        with the same multiplicity.

        NOTE: RandAugment is now applied (skip_autoaugment=False) with paired spatial ops.
        """
        img_list, mask_list = [], []

        for aug_i in range(self.k_tfm):
            if mask0 is None:
                img_list.append(tfm(img0))
            else:
                if self.gaze_sanity and aug_i == 0:
                    img_t, mask_t, dbg_img_pil, dbg_mask_pil = _apply_pair_transform(
                        tfm, img0, mask0, skip_autoaugment=False, return_pil=True
                    )
                    if item is not None:
                        self._save_gaze_sanity(dbg_img_pil, dbg_mask_pil, item, aug_i=aug_i)
                else:
                    img_t, mask_t = _apply_pair_transform(tfm, img0, mask0, skip_autoaugment=False)
                img_list.append(img_t)
                mask_list.append(mask_t)

                # ---- DEBUG ----
                if self.gaze_debug and (self._gaze_dbg_cnt <= self.gaze_debug_max) and aug_i == 0:
                    m = mask_t
                    m_sum = float(m.sum().item())
                    m_min = float(m.min().item())
                    m_max = float(m.max().item())
                    all_zero = (m_sum <= 1e-6)
                    print(f"[GAZE_DEBUG] after_transform tag={debug_tag} aug={aug_i} "
                          f"mask_t shape={tuple(m.shape)} min/max/sum={m_min:.6f}/{m_max:.6f}/{m_sum:.6f} "
                          f"all_zero={all_zero}")

        img = img_list[0] if len(img_list) == 1 else img_list
        if mask0 is None:
            return img, None

        mask = mask_list[0] if len(mask_list) == 1 else mask_list
        return img, mask



