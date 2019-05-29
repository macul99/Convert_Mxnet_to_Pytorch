import torch
import mmdnn
from six import text_type as _text_type
import imp
import sys
from pytorchEmitter import MyPytorchEmitter

'''
for layer in emitter.IR_graph.topological_sort:
    current_node = emitter.IR_graph.get_node(layer)
    node_type = current_node.type
    print('node_type: ', node_type, 'node_name: ', current_node.name)
    if hasattr(emitter, "emit_" + node_type):
        func = getattr(emitter, "emit_" + node_type)
        line = func(current_node)
        if line:
            emitter.add_body(2, line)
    else:
        print("Pytorch Emitter has not supported operator [%s]." % (node_type))
        emitter.emit_UNKNOWN(current_node)
'''


def _get_parser():
    import argparse

    parser = argparse.ArgumentParser(description = 'Convert IR model file formats to other format.')

    parser.add_argument(
        '--phase',
        type=_text_type,
        choices=['train', 'test'],
        default='test',
        help='Convert phase (train/test) for destination toolkits.'
    )

    parser.add_argument(
        '--IRModelPath', '-n', '-in',
        type=_text_type,
        required=True,
        help='Path to the IR network structure file.')

    parser.add_argument(
        '--IRWeightPath', '-w', '-iw',
        type=_text_type,
        required=False,
        default=None,
        help = 'Path to the IR network structure file.')

    parser.add_argument(
        '--dstModelPath', '-d', '-o',
        type = _text_type,
        required = True,
        help = 'Path to save the destination model')

    parser.add_argument(
        '--dstWeightPath', '-dw', '-ow',
        type=_text_type,
        default=None,
        help='Path to save the destination weight.')

    parser.add_argument(
        '--dstPytorchPath', '-dpth',
        type=_text_type,
        default=None,
        help='Path to save the pth file for pytorch.')

    parser.add_argument(
        '--ctx',
        type=_text_type,
        default="gpu",
        help='Path to save the pth file for pytorch.')
    return parser

def convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data

if __name__ == '__main__':

    parser=_get_parser()
    args = parser.parse_args()

    if not args.dstWeightPath or not args.IRWeightPath:
        raise ValueError("Need to set a target weight filename.")

    emitter = MyPytorchEmitter((args.IRModelPath, args.IRWeightPath), ctx=args.ctx)

    #emitter.weights_dict = convert(emitter.weights_dict) # don't know why the dic key is binary string, need to convert to normal string

    emitter.run(args.dstModelPath, args.dstWeightPath, args.phase)

    MainModel = imp.load_source('MainModel', args.dstModelPath)
    model = MainModel.KitModel(args.dstWeightPath)
    model.eval()
    torch.save(model, args.dstPytorchPath)
    #torch.save(model.state_dict(), args.dstPytorchPath) # only save learnable parameters
    print('PyTorch model file is saved as [{}], generated by [{}.py] and [{}].'.format(
          args.dstPytorchPath, args.dstModelPath, args.dstWeightPath))

'''
## first prepare mxnet model for model conversion
# activate mxnet vent
import sys
import numpy as np
sys.path.append('/home/macul/Documents/macul/mklib/utils/')
from mxFaceFeatureExtract import mxFaceFeatureExtract
extractor=mxFaceFeatureExtract('/media/macul/black/mxnet_training/mobilefacenet/dgx_train2','train_2',9, mean_value='/media/macul/black/face_database_raw_data/mscelb_from_insightface/mean.json',mmdnn_convert=True)
extractor.saveModel('./','test_9')

python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n test_9-symbol.json -w test_9-0000.params -d test_9 --inputShape 3,112,112

# now activate pytorch venv
python3 -m convertIR2Pytorch --IRModelPath test_9.pb --dstModelPath net_pytorch_cpu.py --IRWeightPath test_9.npy --dstWeightPath wt_pytorch_cpu.npy --dstPytorchPath pytorch_cpu.pth --ctx cpu
python3 -m convertIR2Pytorch --IRModelPath test_9.pb --dstModelPath net_pytorch_gpu.py --IRWeightPath test_9.npy --dstWeightPath wt_pytorch_gpu.npy --dstPytorchPath pytorch_gpu.pth --ctx gpu

'''