
import torch
import numpy as np
import imageio

from PIL import Image, ImageDraw


def lin2img(tensor, img_res):  # same with the one in plots.py
    batch_size, _, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def gif_rendering(rgb_in, images, path_results, img_res, epoch, fps=6):
    # test
    image = (rgb_in + 1.) / 2.
    image = lin2img(image.unsqueeze(0), img_res).squeeze().numpy().transpose(1, 2, 0)
    image = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), f"epoch: {epoch}", fill=(255, 0, 0))
    images.append(np.array(image))

    # output_file = f"{path_results}/renderings.gif"
    output_file = f"{path_results}/renderings.mp4"
    imageio.mimsave(output_file, images, fps=fps)

    return images


def gif_mirror_sequence(mirror_sequence_cm, images, path_results, img_res, epoch, fps=6):
    # test
    image = mirror_sequence_cm

    image = Image.fromarray((image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), f"epoch: {epoch}", fill=(255, 0, 0))
    images.append(np.array(image))

    output_file = f"{path_results}/mirror_sequences.mp4"
    imageio.mimsave(output_file, images, fps=fps)

    return images


def gif_num_bounce(bounce, images, path_results, img_res, epoch, fps=6, max_bounce=10):
    # test
    image = np.array(bounce.reshape(img_res) / max_bounce * 255, dtype=np.uint8)

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), f"epoch: {epoch}", fill=(255))
    images.append(np.array(image))

    output_file = f"{path_results}/eval_num_bounce.mp4"
    imageio.mimsave(output_file, images, fps=fps)

    return images


def gif_count_penetration(count_penetration, images, path_results, img_res, epoch, fps=6, max_bounce=10):
    # test
    image = np.array(count_penetration.reshape(img_res) / max_bounce * 255, dtype=np.uint8)

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), f"epoch: {epoch}", fill=(255))
    images.append(np.array(image))

    output_file = f"{path_results}/count_penetration.mp4"
    imageio.mimsave(output_file, images, fps=fps)

    return images



def list2tensor_gif(images_input):
    images = torch.tensor(images_input[0]).permute(2, 0, 1)
    images = images.unsqueeze(0).unsqueeze(0)

    for i in range(1, len(images_input)):
        image_i = torch.tensor(images_input[i]).permute(2, 0, 1)
        image_i = image_i.unsqueeze(0).unsqueeze(0)
        images = torch.cat((images, image_i), dim=1)

    return images


