function create_model(path_model, n_ch_input, n_class, n_feature_init; device=torch_device)
    model = py_unet2d.model.unet_model.UNet2D(n_ch_input, n_class, n_feature_init, false)
    
    model.to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    
    return model
end