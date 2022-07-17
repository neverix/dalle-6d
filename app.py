from PIL import Image
import gradio as gr
import numpy as np
import torch


class MidasDepth(object):
    def __init__(self, model_type="DPT_Large", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval().requires_grad_(False)
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def get_depth(self, image):
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if (image > 1).any():
            image /= 255.
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


def main():
    midas = MidasDepth()
    interface = gr.Interface(fn=lambda x: [Image.fromarray(midas.get_depth(x[0]).astype("uint8")), ""], inputs=[
        gr.inputs.Image(),
        gr.inputs.Text()
    ], outputs=[
        gr.outputs.Image(),
        gr.outputs.Video()
    ], title="DALL·E 6D", description="Lift DALL·E 2 (or any other model) into 3D!")
    interface.launch()


if __name__ == '__main__':
    main()
