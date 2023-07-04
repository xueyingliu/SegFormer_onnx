import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor

from torchvision.transforms import Resize 
import torch
import onnx
import onnxruntime as ort
import numpy as np
import copy

def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    pipe = cfg.data.test.pipeline[1:]
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    torch_resize = Resize([512,512])
    input_data = torch_resize(data['img'][0])
    with torch.no_grad():
        # keys = data.type
        # input_data = **data
        model = model.eval()
        # torch.onnx.export(
        #     model,
        #     data['img'][0],
        #     "segformer.b1.512x512.ade.160k.onnx",
        #     verbose=True, 
        #     input_names=['img'], 
        #     output_names=['logits'])
        # 导出onnx
        torch.onnx.export(
            model,
            input_data,
            "segformer.b1.512x512.ade.160k.onnx",
            verbose=True, 
            input_names=['img'], 
            output_names=['logits'],
            opset_version=11)

        # 加载ONNX模型
        onnx_model = onnx.load('segformer.b1.512x512.ade.160k.onnx')
        ort_session = ort.InferenceSession('segformer.b1.512x512.ade.160k.onnx', providers=['CUDAExecutionProvider'])
        ort_img_cpu = input_data.cpu()
        # max_cpu, min_cpu = torch.max(ort_img_cpu), torch.min(ort_img_cpu)
        ort_img_np = np.around(ort_img_cpu.numpy(), 4)
        outputs = ort_session.run(None, {'img': ort_img_np})
        outputs[0] =  np.squeeze(outputs[0])
        # input_data = mmcv.imresize(data['img'][0], (512,512))
        result = model(return_loss=False, rescale=True, img=input_data)
        # max_re, min_re = torch.max(result), torch.min(result)
    return outputs


def get_layer_output(model, image):
    ori_output = copy.deepcopy(model.graph.output)
    
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    
    ort_session = ort.InferenceSession(model.SerializeToString())
    
    
    ort_inputs = {}
    
    for i, input_ele in enumerate(ort_session.get_inputs()):
        ort_inputs[input_ele.name] = image
        
    outputs = [x.name for x in ort_session.get_outputs()]
    ort_outs = ort_session.run(outputs, ort_inputs)
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    
    return ort_outs

def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10)):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, palette=palette, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()
