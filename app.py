from tqdm.auto import trange
from PIL import Image
import gradio as gr
import numpy as np
import pyrender
import trimesh
import scipy
import torch
import cv2
import os


class MidasDepth(object):
    def __init__(self, model_type="DPT_Large", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval().requires_grad_(False)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def get_depth(self, image):
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if (image > 1).any():
            image = image.astype("float64") / 255.
        with torch.inference_mode():
            batch = self.transform(image[..., :3]).to(self.device)
            prediction = self.midas(batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.detach().cpu().numpy()


def process_depth(dep):
    depth = dep.copy()
    depth -= depth.min()
    depth /= depth.max()
    depth = 1 / np.clip(depth, 0.2, 1)
    blurred = cv2.medianBlur(depth, 5)  # 9 not available because it requires 8-bit
    maxd = cv2.dilate(blurred, np.ones((3, 3)))
    mind = cv2.erode(blurred, np.ones((3, 3)))
    edges = maxd - mind
    threshold = .05  # Better to have false positives
    pick_edges = edges > threshold
    return depth, pick_edges


def make_mesh(pic, depth, pick_edges):
    faces = []
    im = np.asarray(pic)
    grid = np.mgrid[0:im.shape[0], 0:im.shape[1]].transpose(1, 2, 0
                                                            ).reshape(-1, 2)[..., ::-1]
    flat_grid = grid[:, 1] * im.shape[1] + grid[:, 0]
    positions = np.concatenate(((grid - np.array(im.shape[:-1])[np.newaxis, :]
                                 / 2) / im.shape[1] * 2,
                                depth.flatten()[flat_grid][..., np.newaxis]),
                               axis=-1)
    positions[:, :-1] *= positions[:, -1:]
    positions[:, 1] *= -1
    colors = im.reshape(-1, 3)[flat_grid]

    c = lambda x, y: y * im.shape[1] + x
    for y in trange(im.shape[0]):
        for x in range(im.shape[1]):
            if pick_edges[y, x]:
                continue
            if x > 0 and y > 0:
                faces.append([c(x, y), c(x, y - 1), c(x - 1, y)])
            if x < im.shape[1] - 1 and y < im.shape[0] - 1:
                faces.append([c(x, y), c(x, y + 1), c(x + 1, y)])
    face_colors = np.asarray([colors[i[0]] for i in faces])

    tri_mesh = trimesh.Trimesh(vertices=positions * np.array([1.0, 1.0, -1.0]),
                               faces=faces,
                               face_colors=np.concatenate((face_colors,
                                                           face_colors[..., -1:]
                                                           * 0 + 255),
                                                          axis=-1).reshape(-1, 4),
                               smooth=False,
                               )

    return tri_mesh


def args_to_mat(tx, ty, tz, rx, ry, rz):
    mat = np.eye(4)
    mat[:3, :3] = scipy.spatial.Rotation.from_euler("XYZ", (rx, ry, rz)).as_matrix()
    mat[:3, 3] = tx, ty, tz
    return mat


def render(mesh, mat):
    scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0]))
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2, aspectRatio=1.0)
    scene.add(camera, pose=mat)
    scene.add(mesh)
    r = pyrender.OffscreenRenderer(1024, 1024)
    rgb, d = r.render(scene, pyrender.constants.RenderFlags.FLAT)
    mask = d == 0
    rgb = rgb.copy()
    rgb[mask] = 0
    res = Image.fromarray(np.concatenate((rgb,
                                          ((mask[..., np.newaxis]) == 0)
                                          .astype(np.uint8) * 255), axis=-1))
    return res


def main():
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

    midas = MidasDepth()
    def fn(pic, *args):
        depth, pick_edges = process_depth(midas.get_depth(pic))
        mesh = make_mesh(pic, depth, pick_edges)
        frame = render(mesh, args_to_mat(*args))
        return np.asarray(frame), (255 / np.asarray(depth)).astype(np.uint8), None

    interface = gr.Interface(fn=fn, inputs=[
        gr.inputs.Image(label="src", type="numpy"),
        gr.inputs.Number(label="tx", default=0.0),
        gr.inputs.Number(label="ty", default=0.0),
        gr.inputs.Number(label="tz", default=0.0),
        gr.inputs.Number(label="rx", default=0.0),
        gr.inputs.Number(label="ry", default=0.0),
        gr.inputs.Number(label="rz", default=0.0)
    ], outputs=[
        gr.outputs.Image(type="numpy"),
        gr.outputs.Image(type="numpy"),
        gr.outputs.Video()
    ], title="DALL·E 6D", description="Lift DALL·E 2 (or any other model) into 3D!")
    interface.launch()


if __name__ == '__main__':
    main()
