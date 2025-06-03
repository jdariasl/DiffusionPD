from typing import Optional

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch


XLA_AVAILABLE = False


class Class_DDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        # self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,  # neglected if init_samples is not None
        latent_dim: int = 64,
        init_samples: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 100,
        return_bottleneck: bool = False,
        device: Optional[torch.device] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if init_samples is not None:
            classification = True
            if isinstance(init_samples, torch.Tensor):
                image = init_samples
                image = image.to(device)
            else:
                raise ValueError(
                    f"init_samples should be of type torch.Tensor, but got {type(init_samples)}"
                )
        else:
            classification = False
            image = torch.randn(batch_size, latent_dim, generator=generator).to(device)

        if class_labels is not None:
            if isinstance(class_labels, torch.Tensor):
                label = class_labels.to(device)
            else:
                raise ValueError(
                    f"class_labels should be of type torch.Tensor, but got {type(class_labels)}"
                )
        else:
            label = torch.randint(0, 4, (batch_size,), device=device)

        if classification:
            model = self.unet.to(device)
            last_t = self.scheduler.timesteps[-2]
            for t in self.scheduler.timesteps[-num_inference_steps:]:
                # 1. predict noise model_output
                t2 = t * torch.ones(image.shape[0], dtype=torch.int64)

                model_output = model(image.to(device), label.to(device), t2.to(device))

                # 2. compute previous image: x_t -> x_t-1
                image = self.scheduler.step(
                    model_output, t, image, generator=generator
                ).prev_sample
                if t == last_t:
                    # if we are at the last step, we return the image
                    if return_bottleneck:
                        t2 = self.scheduler.timesteps[-1] * torch.ones(
                            image.shape[0], dtype=torch.int64
                        )
                        bottleneck, _, _, _, _, _, _ = model.encode(
                            image.to(device), label.to(device), t2.to(device)
                        )
                        model_output = model(
                            image.to(device), label.to(device), t2.to(device)
                        )

                        # 2. compute previous image: x_t -> x_t-1
                        image = self.scheduler.step(
                            model_output, t, image, generator=generator
                        ).prev_sample
                        return image, bottleneck
                else:
                    pass
            return image

        # set step values
        if not classification:
            self.scheduler.set_timesteps(num_inference_steps)

            for t in self.scheduler.timesteps:
                model = self.unet.to(device)
                # 1. predict noise model_output
                t2 = t * torch.ones(image.shape[0], dtype=torch.int64).to(device)
                model_output = model(image.to(device), label.to(device), t2.to(device))

                # 2. compute previous image: x_t -> x_t-1
                image = self.scheduler.step(
                    model_output, t, image, generator=generator
                ).prev_sample

            return image
