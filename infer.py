from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

config_file = 'local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py'
checkpoint_file = 'checkpoints/segformer.b1.512x512.ade.160k.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


# test a single image
img = 'demo/2.jpg'
result = inference_segmentor(model, img)


# show the results
show_result_pyplot(model, img, result, get_palette('ade'))