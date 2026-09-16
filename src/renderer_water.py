import torch
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.field_components.mlp import MLP
from torch import nn


class WaterMediumRenderer(nn.Module):
    def __init__(self, config, ckpt_path):
        super().__init__()
        num_layers_medium = config.get("num_layers_medium", 2)
        hidden_dim_medium = config.get("hidden_dim_medium", 128)
        mlp_type = config.get("mlp_type", "tcnn")

        self.direction_encoding = SHEncoding(levels=4, implementation="tcnn")

        self.colour_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

        if num_layers_medium > 1:
            self.medium_mlp = MLP(
                in_dim=self.direction_encoding.get_out_dim(),
                num_layers=num_layers_medium,
                layer_width=hidden_dim_medium,
                out_dim=9,
                activation=nn.Sigmoid(),
                out_activation=None,
                implementation=mlp_type,
            )
        else:
            self.medium_mlp = nn.Linear(self.direction_encoding.get_out_dim(), 9)

        self._load_weights(ckpt_path)

    def _load_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        state_dict = checkpoint.get("pipeline", checkpoint)

        my_state = self.state_dict()
        new_state = {}

        for k, v in state_dict.items():
            if "medium_mlp" in k or "direction_encoding" in k:
                for my_key in my_state:
                    if (
                        k.endswith(my_key) or my_key.endswith(k.split("_model.")[-1])
                    ) and (my_state[my_key].shape == v.shape):
                        new_state[my_key] = v

        self.load_state_dict(new_state, strict=False)
        print("The TCNN weights for the water environment have been loaded.")

    @torch.no_grad()
    def get_medium_parameters(self, c2w, K, width, height):
        """
        Returns the physical parameters of the water (RGB, backscatter, attenuation)
        as [H, W, 3] tensors for a given camera perspective.
        """
        ys, xs = torch.meshgrid(
            torch.arange(height, device="cuda"),
            torch.arange(width, device="cuda"),
            indexing="ij",
        )

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (xs + 0.5 - cx) / fx
        y = (ys + 0.5 - cy) / fy
        z = torch.ones_like(x)

        directions_camera = torch.stack([x, y, z], dim=-1)
        directions_camera = torch.nn.functional.normalize(directions_camera, dim=-1)

        directions_world = directions_camera @ c2w[:3, :3].T
        directions_flat = directions_world.reshape(-1, 3)

        encoded = self.direction_encoding(directions_flat)
        raw_output = self.medium_mlp(encoded)

        medium_rgb = self.colour_activation(raw_output[..., 0:3]).reshape(
            height, width, 3
        )
        sigma_backscatter = self.sigma_activation(raw_output[..., 3:6]).reshape(
            height, width, 3
        )
        sigma_attenuation = self.sigma_activation(raw_output[..., 6:9]).reshape(
            height, width, 3
        )

        return medium_rgb, sigma_backscatter, sigma_attenuation
