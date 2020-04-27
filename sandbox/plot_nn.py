from src.nn import NeuralNetwork
import cv2 as cv


nn = NeuralNetwork.loader('20200425_064103_id_4')

if nn.h_layers_w is None:
    nlayers = 2
else:
    nlayers = 2 + len(nn.h_layers)

im_w = 1600
im_h = 800

img = np.zeros((im_w, im_h, 2), np.unit8)
buf_x = 50
buf_y = 50
layer_space = int((im_w - buf_x * 2) / nlayers)

for i in range(nn.inputs.shape[0]):
    x =
    cv.circle()
