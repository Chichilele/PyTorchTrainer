import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def plot_loss(results):
    train_losses, train_counter = results["train_loss"]
    test_losses, test_counter = results["test_loss"]

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    return fig


class CustomSubset(Dataset):
    r"""
    Subset of a dataset at specified indices with given transform.
    Same as torch.utils.data.Subset but with transform support.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (Transform): Transform operation to be applied to the dataset.
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)
class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = (output.cpu()).data

    def remove(self):
        self.hook.remove()


def cam(
    model, image, input_shape=(224, 224), last_conv_name="conv5",
):
    """ run and plot CAM on model for image.
    """

    from torchvision import transforms

    ## prepare input image
    img_transform = transforms.Compose(
        [transforms.Resize(input_shape), transforms.ToTensor()]
    )
    transformed_img = img_transform(image)
    activated_features, weights, class_idx = get_activations(
        model, transformed_img, last_conv_name,
    )

    ## compute CAM
    cam_features = compute_cam(activated_features, weights, class_idx)

    ## plot image and overlay
    overlay_transform = transforms.Compose(
        [transforms.Resize(input_shape, interpolation=0)]  # Image.NEAREST
    )
    transformed_overlay = overlay_transform(cam_features[None, :])[0]
    plot_cam(transformed_img, transformed_overlay)


def get_activations(model, image, last_conv_name):
    """Get conv and fc layer's activations as well as predicted class
    """

    from torch import topk

    ## get last conv layer's weights
    final_layer = model._modules.get(last_conv_name)
    activated_features = SaveFeatures(final_layer)

    ## get last conv layer's activation
    prediction = model(image[None, :, :].cuda())
    activated_features.remove()

    ## get fc layer's weights
    weight_softmax_params = list(model._modules.get("classifier").parameters())
    weight_softmax = weight_softmax_params[0].cpu().data.squeeze()

    ## get predicted class index
    pred_probabilities = F.softmax(prediction).data.squeeze()
    class_idx = topk(pred_probabilities, 1)[1].item()

    return activated_features.features, weight_softmax, class_idx


def compute_cam(feature_conv, weight_fc, class_idx):
    """Compute the class activation map for a given 
    conv layer and FC layer on a class index.
    """

    from torch import mm

    _, nc, h, w = feature_conv.shape
    cam = mm(weight_fc[class_idx][None, :], feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - cam.min()
    cam_img = cam / cam.max()
    return cam_img


def plot_cam(transformed_img, transformed_overlay):
    """plot the CAM 
    """

    from matplotlib.pyplot import imshow, show

    imshow(transformed_img.permute(1, 2, 0))
    imshow(transformed_overlay, alpha=0.5, cmap="jet")
    show()
