import segmentation_models_pytorch as smp


def build_unet():
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=4
    )

    return model
