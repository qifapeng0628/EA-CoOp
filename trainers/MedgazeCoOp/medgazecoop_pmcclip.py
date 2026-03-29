import copy
import math
import os
import os.path as osp
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.prompt_templates2 import MEDGAZECOOP_TEMPLATES

from clip.pmcclip import ModifiedResNet
from transformers import AutoTokenizer, AutoModel

directory = "clip/checkpoints"

files = {
    "text_encoder.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}


def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")


class TextEncoder(nn.Module):
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts, tokenized_prompts):
        output = self.text_encoder(
            inputs_embeds=prompts.cuda(),
            attention_mask=tokenized_prompts['attention_mask'].cuda()
        )
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        return text_feature


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MEDGAZECOOP.N_CTX
        ctx_init = cfg.TRAINER.MEDGAZECOOP.CTX_INIT
        dtype = torch.float32
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx == 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(
                ctx_init, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
            )['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.MEDGAZECOOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]

        # only used when CLASS_TOKEN_POSITION is middle/front
        name_lens = []
        for name in classnames:
            encoded = self.tokenizer(
                name, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
            )
            attn_len = int(encoded['attention_mask'][0].sum().item())
            name_lens.append(max(attn_len - 2, 0))

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = self.tokenizer(
            prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
        )

        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(
                tokenized_prompts['input_ids'].cuda()
            ).type(dtype)

            all_teacher_features = []
            for i in range(cfg.TRAINER.MEDGAZECOOP.N_PROMPTS):
                x_tokenized = self.tokenizer(
                    [MEDGAZECOOP_TEMPLATES[classname][i] for classname in classnames],
                    padding='max_length',
                    truncation=True,
                    max_length=77,
                    return_tensors='pt'
                )
                text_features = pmcclip_model.text_encoder(
                    x_tokenized['input_ids'].cuda(),
                    attention_mask=x_tokenized['attention_mask'].cuda()
                )
                pooler_output = text_features.pooler_output
                text_features = pooler_output @ pmcclip_model.text_projection_layer
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MEDGAZECOOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


class PMCCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')

    def forward(self, image, text):
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']
        return image_feature, text_feature


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, pmcclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = pmcclip_model.image_encoder
        self.text_encoder = TextEncoder(pmcclip_model)
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        self.gaze_cons_lambda = float(os.getenv("GAZE_CONS_LAMBDA", "0.0"))
        self.gaze_cons_temp = float(os.getenv("GAZE_CONS_T", os.getenv("GAZE_CONS_TEMP", "1.0")))
        self.gaze_roi_ce_lambda = float(os.getenv("GAZE_ROI_CE_LAMBDA", "0.0"))
        self.gaze_scgta_lambda = float(os.getenv("GAZE_SCGTA_LAMBDA", "0.0"))
        self.gaze_scgta_temp = float(os.getenv("GAZE_SCGTA_T", os.getenv("GAZE_SCGTA_TEMP", "0.1")))

    def forward(self, image, label=None, crop_img=None, crop_valid=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = math.exp(self.logit_scale)

        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        if isinstance(image_features, dict):
            image_features = image_features['image_features']
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            loss_ce = F.cross_entropy(logits, label)

            loss_cons = torch.tensor(0.0, device=logits.device)
            loss_roi_ce = torch.tensor(0.0, device=logits.device)
            loss_gaze_scgta = torch.tensor(0.0, device=logits.device)

            need_crop = (crop_img is not None) and (
                (self.gaze_cons_lambda > 0) or (self.gaze_roi_ce_lambda > 0) or (self.gaze_scgta_lambda > 0)
            )
            if need_crop:
                valid = crop_valid
                if valid is None:
                    valid = torch.ones(crop_img.shape[0], dtype=torch.bool, device=crop_img.device)

                if valid.any():
                    crop_features = self.image_encoder(crop_img.type(self.dtype))
                    if isinstance(crop_features, dict):
                        crop_features = crop_features['image_features']
                    crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
                    logits_crop = logit_scale * crop_features @ text_features.t()

                    if self.gaze_roi_ce_lambda > 0:
                        ce_per = F.cross_entropy(logits_crop, label, reduction='none')
                        w = valid.to(ce_per.dtype)
                        denom = torch.clamp(w.sum(), min=1.0)
                        loss_roi_ce = (ce_per * w).sum() / denom
                        loss_roi_ce = loss_roi_ce * self.gaze_roi_ce_lambda

                    if self.gaze_cons_lambda > 0:
                        T = float(self.gaze_cons_temp)
                        idx = valid.nonzero(as_tuple=False).squeeze(1)
                        p = F.softmax(logits_crop[idx] / T, dim=1).detach()
                        log_q = F.log_softmax(logits[idx] / T, dim=1)
                        loss_cons = F.kl_div(log_q, p, reduction='batchmean') * (T * T)
                        loss_cons = loss_cons * self.gaze_cons_lambda

                    if self.gaze_scgta_lambda > 0:
                        teacher = self.prompt_learner.fixed_embeddings
                        teacher = teacher.to(device=crop_features.device, dtype=crop_features.dtype)
                        teacher_norm = teacher / teacher.norm(dim=-1, keepdim=True)

                        idx = valid.nonzero(as_tuple=False).squeeze(1)
                        y = label[idx]
                        v = crop_features[idx]
                        T_s = float(self.gaze_scgta_temp)

                        t_hat = torch.zeros((v.size(0), teacher_norm.size(-1)), device=v.device, dtype=v.dtype)
                        for ii in range(v.size(0)):
                            c = int(y[ii].item())
                            sims = teacher_norm[c] @ v[ii]
                            T_s_eff = T_s if T_s > 0 else 1e-6
                            w = F.softmax((sims - sims.max()) / T_s_eff, dim=0)
                            t = (w.unsqueeze(0) @ teacher_norm[c]).squeeze(0)
                            t = t / (t.norm() + 1e-6)
                            t_hat[ii] = t

                        uniq = torch.unique(y)
                        per_class_losses = []
                        for c in uniq.tolist():
                            mask = (y == c)
                            if not mask.any():
                                continue
                            t_bar = t_hat[mask].mean(dim=0)
                            t_bar = t_bar / (t_bar.norm() + 1e-6)
                            p_cls = text_features[int(c)]
                            per_class_losses.append(
                                F.mse_loss(p_cls.float(), t_bar.detach().float(), reduction='mean')
                            )

                        if len(per_class_losses) > 0:
                            loss_gaze_scgta = torch.stack(per_class_losses).mean() * self.gaze_scgta_lambda

            return logits, loss_ce, loss_cons, loss_roi_ce, loss_gaze_scgta
        else:
            return logits


@TRAINER_REGISTRY.register()
class MedgazeCoOp_PMCCLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MEDGAZECOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        for filename, url in files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print("Loading PMC-CLIP (backbone: RN50)")
        image_encoder = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory, 'image_encoder(resnet50).pth'), weights_only=True))

        text_encoder = AutoModel.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory, 'text_encoder.pth'), weights_only=True))

        text_projection_layer = torch.load(os.path.join(directory, 'text_projection_layer.pth'), weights_only=True)
        text_projection_layer = nn.Parameter(text_projection_layer)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()

        print("Building custom CLIP (MedgazeCoOp)")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner.ctx"]
        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.MEDGAZECOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label, crop_img, crop_valid = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MEDGAZECOOP.PREC
        if prec == "amp":
            with autocast():
                logits, loss_ce, loss_cons, loss_roi_ce, loss_gaze_scgta = model(
                    image, label, crop_img=crop_img, crop_valid=crop_valid
                )
                loss = loss_ce + loss_cons + loss_roi_ce + loss_gaze_scgta
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_ce, loss_cons, loss_roi_ce, loss_gaze_scgta = model(
                image, label, crop_img=crop_img, crop_valid=crop_valid
            )
            loss = loss_ce + loss_cons + loss_roi_ce + loss_gaze_scgta
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
            "loss_cons": loss_cons.item() if torch.is_tensor(loss_cons) else float(loss_cons),
            "loss_roi_ce": loss_roi_ce.item() if torch.is_tensor(loss_roi_ce) else float(loss_roi_ce),
            "loss_gaze_scgta": loss_gaze_scgta.item() if torch.is_tensor(loss_gaze_scgta) else float(loss_gaze_scgta),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        crop_img = batch.get("crop_img", None)
        crop_valid = batch.get("crop_valid", None)

        if isinstance(input, (list, tuple)):
            input = input[0]
        if isinstance(crop_img, (list, tuple)):
            crop_img = crop_img[0]

        input = input.to(self.device)
        label = label.to(self.device)
        if crop_img is not None:
            crop_img = crop_img.to(self.device)
        if crop_valid is not None:
            crop_valid = crop_valid.to(self.device)
        return input, label, crop_img, crop_valid

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} ".format(name) + 'from "{}" (epoch = {})'.format(model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
