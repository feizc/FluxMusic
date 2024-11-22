from model import FluxParams, Flux

def build_model(version='base'):
    if version == 'base': 
        params=FluxParams(
            in_channels=32,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=768,
            mlp_ratio=4.0,
            num_heads=16,
            depth=12,
            depth_single_blocks=24,
            axes_dim=[16, 16, 16],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ) 
    
    elif version == 'small': 
        params=FluxParams(
            in_channels=32,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=512,
            mlp_ratio=4.0,
            num_heads=16,
            depth=8,
            depth_single_blocks=16,
            axes_dim=[8, 12, 12],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ) 
        

    elif version == 'large': 
        params=FluxParams(
            in_channels=32,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=1024,
            mlp_ratio=4.0,
            num_heads=16,
            depth=12,
            depth_single_blocks=24,
            axes_dim=[16, 24, 24],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ) 

    elif version == 'biggiant': 
        params=FluxParams(
            in_channels=32,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=2048,
            mlp_ratio=4.0,
            num_heads=16,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[32, 48, 48],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ) 

    elif version == 'giant_full':    
        params=FluxParams(
            in_channels=32,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=1408,
            mlp_ratio=4.0,
            num_heads=16,
            depth=12,
            depth_single_blocks=24,
            axes_dim=[16, 36, 36],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        )

    else:     
        params=FluxParams(
            in_channels=32,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=1408,
            mlp_ratio=4.0,
            num_heads=16,
            depth=16,
            depth_single_blocks=32,
            axes_dim=[16, 36, 36],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        )

        
    model = Flux(params) 
    return model 