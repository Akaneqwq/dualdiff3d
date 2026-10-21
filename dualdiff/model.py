import os
SD_TURBO_PATH = os.environ.get('SD_TURBO_PATH', 'stabilityai/sd-turbo')
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from einops import rearrange, repeat
try:
    from dualdiff.attention import ReferenceAttentionControl
except ImportError:
    from attention import ReferenceAttentionControl


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained(SD_TURBO_PATH, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def load_ckpt_from_state_dict(net, optimizer, pretrained_path):
    sd = torch.load(
        pretrained_path, map_location="cpu", mmap=True, weights_only=True
    )

    if "state_dict_vae" in sd:
        _sd_vae = net.vae.state_dict()
        for k in sd["state_dict_vae"]:
            _sd_vae[k] = sd["state_dict_vae"][k]
        net.vae.load_state_dict(_sd_vae)
    _sd_unet = net.unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    net.unet.load_state_dict(_sd_unet)

    if "state_dict_ref_unet" in sd:
        _sd_ref_unet = net.ref_unet.state_dict()
        for k in sd["state_dict_ref_unet"]:
            _sd_ref_unet[k] = sd["state_dict_ref_unet"][k]
        net.ref_unet.load_state_dict(_sd_ref_unet)

    optimizer.load_state_dict(sd["optimizer"])

    return net, optimizer


def save_ckpt(net, optimizer, outf):
    sd = {}
    sd["vae_lora_target_modules"] = net.target_modules_vae
    sd["rank_vae"] = net.lora_rank_vae
    sd["state_dict_unet"] = net.unet.state_dict()
    sd["state_dict_vae"] = {k: v for k, v in net.vae.state_dict().items() if "lora" in k or "skip" in k}
    sd["state_dict_ref_unet"] = net.ref_unet.state_dict()

    sd["optimizer"] = optimizer.state_dict()

    torch.save(sd, outf)


class DualDiff(torch.nn.Module):
    """Dual structure-appearance diffusion model for novel-view refinement.

    The structure branch (``unet``) consumes the degraded rendering and owns the
    VAE decoder; the appearance branch (``ref_unet``) consumes the reference view
    and exposes its self-attention hidden states through Structure-Appearance
    Attention (see ``attention.ReferenceAttentionControl``).

    Passing ``pretrained_path`` loads DualDiff weights; leaving it as ``None``
    starts from SD-Turbo with a freshly initialized VAE LoRA and skip convs.
    """

    def __init__(self, pretrained_path=None, lora_rank_vae=4, timestep=199):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(SD_TURBO_PATH, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(SD_TURBO_PATH, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained(SD_TURBO_PATH, subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False

        unet = UNet2DConditionModel.from_pretrained(SD_TURBO_PATH, subfolder="unet")
        ref_unet = UNet2DConditionModel.from_pretrained(SD_TURBO_PATH, subfolder="unet")

        if pretrained_path is not None:
            sd = torch.load(
                pretrained_path, map_location="cpu", mmap=True, weights_only=True
            )
            vae.encoder.requires_grad_(False)
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            # Keep the LoRA configuration on the instance for checkpoint saving.
            self.lora_rank_vae = sd["rank_vae"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            _sd_ref_unet = ref_unet.state_dict()
            for k in sd["state_dict_ref_unet"]:
                _sd_ref_unet[k] = sd["state_dict_ref_unet"][k]
            ref_unet.load_state_dict(_sd_ref_unet)

        else:
            print("No DualDiff checkpoint given; initializing VAE LoRA and skip convs from scratch")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            
            target_modules = []
            for id, (name, param) in enumerate(vae.named_modules()):
                if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                    target_modules.append(name)
            target_modules_vae = target_modules
            vae.encoder.requires_grad_(False)

            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
                
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae

        unet.to("cuda")
        vae.to("cuda")
        ref_unet.to("cuda")

        self.unet, self.vae, self.ref_unet = unet, vae, ref_unet
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.sched.timesteps = self.timesteps 
        
        self.text_encoder.requires_grad_(False)
        
        self.reference_control_writer = ReferenceAttentionControl(
            self.ref_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.ref_unet.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.ref_unet.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.ref_unet.train()
        self.unet.requires_grad_(True)

        for n, _p in self.ref_unet.named_parameters():
            if "up_blocks.3" in n:
                _p.requires_grad_(False)
            else:
                _p.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
        
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in Ref UNet: {sum(p.numel() for p in self.ref_unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def _resolve_timesteps(self, timesteps, device):
        if timesteps is None:
            return self.timesteps.to(device=device)
        resolved = torch.as_tensor(timesteps, device=device, dtype=torch.long).reshape(-1)
        if resolved.numel() != 1:
            raise ValueError("timesteps must be a scalar or contain exactly one value")
        return resolved

    @staticmethod
    def _expand_caption_for_views(caption_enc, batch_size, num_views):
        expected_batch = batch_size * num_views
        if caption_enc.shape[0] == expected_batch:
            return caption_enc
        if caption_enc.shape[0] != batch_size:
            raise ValueError(
                "Text embedding batch must match either the input batch or "
                f"the flattened view batch: {caption_enc.shape[0]} not in "
                f"{{{batch_size}, {expected_batch}}}"
            )
        return repeat(caption_enc, 'b n c -> (b v) n c', v=num_views)

    def forward(self, x, ref_x=None, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        num_views = x.shape[1]
        batch_size = x.shape[0]
        active_timesteps = self._resolve_timesteps(timesteps, x.device)

        if prompt is not None:
            # The same prompt is reused for every sample in the batch.
            prompt_list = [prompt] * batch_size
            caption_tokens = self.tokenizer(
                prompt_list,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]  # [batch_size, seq_len, hidden_dim]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        caption_enc = self._expand_caption_for_views(
            caption_enc, batch_size, num_views
        )

        # Encode the reference first so the input encode below determines the
        # decoder skip activations, matching the training implementation.
        if ref_x is not None:
            if ref_x.shape[:2] != (batch_size, num_views):
                raise ValueError("Reference and input must have the same batch and view counts")
            ref_x = rearrange(ref_x, 'b v c h w -> (b v) c h w')
            ref_z = self.vae.encode(ref_x).latent_dist.sample() * self.vae.config.scaling_factor 
            self.ref_unet(
                ref_z,
                active_timesteps,
                encoder_hidden_states=caption_enc,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        x = rearrange(x, 'b v c h w -> (b v) c h w')
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        input_skip_acts = self.vae.encoder.current_down_blocks

        model_pred = self.unet(z, active_timesteps, encoder_hidden_states=caption_enc,).sample
        self.sched.timesteps = active_timesteps
        z_denoised = self.sched.step(model_pred, active_timesteps, z, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = input_skip_acts
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)
        
        self.reference_control_reader.clear()
        self.reference_control_writer.clear()
        return output_image

    def sample_batch(self, images, width, height, ref_images=None, timesteps=None, prompt=None, prompt_tokens=None):
        """Refine a batch of renderings.

        Args:
            images: List[PIL.Image] of length B, the degraded renderings.
            width: target width fed to the network.
            height: target height fed to the network.
            ref_images: Optional[List[PIL.Image]] of length B, the reference views.
            timesteps: Optional scalar timestep overriding the configured value.
            prompt: Optional[str] shared by every sample.
            prompt_tokens: Optional[torch.Tensor] of pre-encoded text tokens.
        Returns:
            List[PIL.Image] of length B, resized back to each input resolution.
        """
        batch_size = len(images)

        original_sizes = [img.size for img in images]  # [(width, height), ...]

        processed_tensors = []
        for img in images:
            # The VAE requires side lengths that are multiples of 8.
            new_width = img.width - img.width % 8
            new_height = img.height - img.height % 8
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            T = transforms.Compose([
                transforms.Resize((height, width), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            tensor = T(img_resized)  # [C, H, W]
            processed_tensors.append(tensor)

        x = torch.stack(processed_tensors, dim=0).unsqueeze(1).cuda()  # [B, 1, C, H, W]

        if ref_images is not None:
            if len(ref_images) != batch_size:
                raise ValueError(
                    f"Got {len(ref_images)} reference images for {batch_size} inputs; counts must match"
                )

            ref_tensors = []
            for ref_img in ref_images:
                new_width = ref_img.width - ref_img.width % 8
                new_height = ref_img.height - ref_img.height % 8
                ref_resized = ref_img.resize((new_width, new_height), Image.LANCZOS)
                
                T = transforms.Compose([
                    transforms.Resize((height, width), interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                ref_tensor = T(ref_resized)
                ref_tensors.append(ref_tensor)
            
            ref_x = torch.stack(ref_tensors, dim=0).unsqueeze(1).cuda()  # [B, 1, C, H, W]
        else:
            ref_x = None
        
        output_tensors = self.forward(x, ref_x, timesteps, prompt, prompt_tokens)
        
        output_images = []
        for i in range(batch_size):
            output_tensor = output_tensors[i, 0]  # sample i, view 0: [C, H, W]

            # Undo the [-1, 1] normalization.
            output_tensor = output_tensor.cpu() * 0.5 + 0.5
            output_tensor = torch.clamp(output_tensor, 0, 1)

            output_pil = transforms.ToPILImage()(output_tensor)

            orig_width, orig_height = original_sizes[i]
            output_pil = output_pil.resize((orig_width, orig_height), Image.LANCZOS)

            output_images.append(output_pil)

        return output_images

    def sample_batch_with_seeds(
        self,
        images,
        width,
        height,
        ref_images=None,
        timesteps=None,
        prompt=None,
        prompt_tokens=None,
        seeds=None,
    ):
        """Refine a batch of renderings with one random seed per sample.

        Used by the RRO loop to draw K independent repairs of the same view.

        Args:
            images: List[PIL.Image] of length B.
            seeds: List[int] of length B, one seed per sample.
        """
        batch_size = len(images)
        if seeds is None or len(seeds) != batch_size:
            raise ValueError("seeds must be provided and match batch size")

        original_sizes = [img.size for img in images]

        processed_tensors = []
        for img in images:
            new_width = img.width - img.width % 8
            new_height = img.height - img.height % 8
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)

            T = transforms.Compose([
                transforms.Resize((height, width), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            tensor = T(img_resized)
            processed_tensors.append(tensor)

        x = torch.stack(processed_tensors, dim=0).unsqueeze(1).cuda()

        if ref_images is not None:
            if len(ref_images) != batch_size:
                raise ValueError(
                    f"Got {len(ref_images)} reference images for {batch_size} inputs; counts must match"
                )

            ref_tensors = []
            for ref_img in ref_images:
                new_width = ref_img.width - ref_img.width % 8
                new_height = ref_img.height - ref_img.height % 8
                ref_resized = ref_img.resize((new_width, new_height), Image.LANCZOS)

                T = transforms.Compose([
                    transforms.Resize((height, width), interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                ref_tensor = T(ref_resized)
                ref_tensors.append(ref_tensor)

            ref_x = torch.stack(ref_tensors, dim=0).unsqueeze(1).cuda()
        else:
            ref_x = None

        output_tensors = self._forward_with_seeds(x, ref_x, timesteps, prompt, prompt_tokens, seeds)

        output_images = []
        for i in range(batch_size):
            output_tensor = output_tensors[i, 0]
            output_tensor = output_tensor.cpu() * 0.5 + 0.5
            output_tensor = torch.clamp(output_tensor, 0, 1)
            output_pil = transforms.ToPILImage()(output_tensor)
            orig_width, orig_height = original_sizes[i]
            output_pil = output_pil.resize((orig_width, orig_height), Image.LANCZOS)
            output_images.append(output_pil)

        return output_images

    def _forward_with_seeds(self, x, ref_x, timesteps, prompt, prompt_tokens, seeds):
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        num_views = x.shape[1]
        batch_size = x.shape[0]
        active_timesteps = self._resolve_timesteps(timesteps, x.device)

        if num_views > 1:
            expanded_seeds = []
            for s in seeds:
                expanded_seeds.extend([s] * num_views)
        else:
            expanded_seeds = seeds

        if prompt is not None:
            prompt_list = [prompt] * batch_size
            caption_tokens = self.tokenizer(
                prompt_list,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        caption_enc = self._expand_caption_for_views(
            caption_enc, batch_size, num_views
        )

        if ref_x is not None:
            if ref_x.shape[:2] != (batch_size, num_views):
                raise ValueError("Reference and input must have the same batch and view counts")
            ref_x = rearrange(ref_x, 'b v c h w -> (b v) c h w')
            ref_latent_dist = self.vae.encode(ref_x).latent_dist
            ref_mean = ref_latent_dist.mean
            ref_logvar = ref_latent_dist.logvar
            ref_std = torch.exp(0.5 * ref_logvar)

            ref_noises = []
            for i, seed in enumerate(expanded_seeds):
                g = torch.Generator(device=ref_mean.device).manual_seed(int(seed))
                ref_noises.append(torch.randn(ref_mean[i].shape, device=ref_mean.device, generator=g))
            ref_noise = torch.stack(ref_noises, dim=0)
            ref_z = (ref_mean + ref_std * ref_noise) * self.vae.config.scaling_factor

            self.ref_unet(
                ref_z,
                active_timesteps,
                encoder_hidden_states=caption_enc,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        x = rearrange(x, 'b v c h w -> (b v) c h w')
        latent_dist = self.vae.encode(x).latent_dist
        input_skip_acts = self.vae.encoder.current_down_blocks
        mean = latent_dist.mean
        logvar = latent_dist.logvar
        std = torch.exp(0.5 * logvar)

        noises = []
        for i, seed in enumerate(expanded_seeds):
            g = torch.Generator(device=mean.device).manual_seed(int(seed))
            noises.append(torch.randn(mean[i].shape, device=mean.device, generator=g))
        noise = torch.stack(noises, dim=0)
        z = (mean + std * noise) * self.vae.config.scaling_factor

        model_pred = self.unet(z, active_timesteps, encoder_hidden_states=caption_enc,).sample
        self.sched.timesteps = active_timesteps
        z_denoised = self.sched.step(model_pred, active_timesteps, z, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = input_skip_acts
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=num_views)

        self.reference_control_reader.clear()
        self.reference_control_writer.clear()
        return output_image

    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        input_width, input_height = image.size
        new_width = image.width - image.width % 8
        new_height = image.height - image.height % 8
        image = image.resize((new_width, new_height), Image.LANCZOS)
        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        if ref_image is None:
            ref_x = None
        else:
            ref_image = ref_image.resize((new_width, new_height), Image.LANCZOS)
            ref_x = T(ref_image).unsqueeze(0).unsqueeze(0).cuda()
        
        output_image = self.forward(x, ref_x, timesteps, prompt, prompt_tokens)[:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_width, input_height), Image.LANCZOS)
        
        return output_pil
