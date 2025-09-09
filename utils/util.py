import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import errno
import cv2

class ActivationsAndGradients:
  """ Class for extracting activations and
  registering gradients from targeted intermediate layers """

  def __init__(self, model, target_layers, reshape_transform):
      self.model = model
      self.gradients = []
      self.activations = []
      self.reshape_transform = reshape_transform
      self.handles = []
      for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
          # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

  def save_activation(self, module, input, output):
      activation = output
      if self.reshape_transform is not None:
          activation = self.reshape_transform(activation)
      self.activations.append(activation.cpu().detach())

  def save_gradient(self, module, grad_input, grad_output):
      # Gradients are computed in reverse order
      grad = grad_output[0]
      if self.reshape_transform is not None:
          grad = self.reshape_transform(grad)
      self.gradients = [grad.cpu().detach()] + self.gradients

  def __call__(self, x):
      self.gradients = []
      self.activations = []
      return self.model(x, x, modal=0)

  def release(self):
      for handle in self.handles:
          handle.remove()

class GradCAM:
   def __init__(self, model, target_layers, reshape_transform=None):
       self.model = model.train()
       self.target_layers = target_layers
       self.reshape_transform = reshape_transform
       self.activations_and_grads = ActivationsAndGradients(
           self.model, target_layers, reshape_transform)

   """ Get a vector of weights for every channel in the target layer.
      Methods that return weights channels,
      will typically need to only implement this function. """

   @staticmethod
   def get_cam_weights(grads):
       return np.mean(grads, axis=(2, 3), keepdims=True)

   @staticmethod
   def get_loss(output, target_category):
       loss = 0
       for i in range(len(target_category)):
           loss = loss + output[i, target_category[i]]
       return loss

   def get_cam_image(self, activations, grads):
       weights = self.get_cam_weights(grads)
       weighted_activations = weights * activations
       cam = weighted_activations.sum(axis=1)

       return cam
   

   @staticmethod
   def get_target_width_height(input_tensor):
       width, height = input_tensor.size(-1), input_tensor.size(-2)
       return width, height

   def compute_cam_per_layer(self, input_tensor):
       activations_list = [a.cpu().data.numpy()
                           for a in self.activations_and_grads.activations]
       grads_list = [g.cpu().data.numpy()
                     for g in self.activations_and_grads.gradients]
       target_size = self.get_target_width_height(input_tensor)

       cam_per_target_layer = []
       # Loop over the saliency image from every layer

       for layer_activations, layer_grads in zip(activations_list, grads_list):
           cam = self.get_cam_image(layer_activations, layer_grads)
           cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
           scaled = self.scale_cam_image(cam, target_size)
           cam_per_target_layer.append(scaled[:, None, :])

       return cam_per_target_layer

   def aggregate_multi_layers(self, cam_per_target_layer):
       cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
       cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
       result = np.mean(cam_per_target_layer, axis=1)
       return self.scale_cam_image(result)

   @staticmethod
   def scale_cam_image(cam, target_size=None):
       result = []
       for img in cam:
           img = img - np.min(img)
           img = img / (1e-7 + np.max(img))
           if target_size is not None:
               img = cv2.resize(img, target_size)
           result.append(img)
       result = np.float32(result)

       return result

   def __call__(self, input_tensor, target_category=None):
       # 正向传播得到网络输出logits(未经过softmax)
       output = self.activations_and_grads(input_tensor)
       if isinstance(target_category, int):
           target_category = [target_category] * input_tensor.size(0)

       if target_category is None:
           target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
           print(f"category id: {target_category}")
       else:
           assert (len(target_category) == input_tensor.size(0))

       self.model.zero_grad()
       loss = self.get_loss(output, target_category)
       loss.backward()

       # In most of the saliency attribution papers, the saliency is
       # computed with a single target layer.
       # Commonly it is the last convolutional layer.
       # Here we support passing a list with multiple target layers.
       # It will compute the saliency image for every image,
       # and then aggregate them (with a default mean aggregation).
       # This gives you more flexibility in case you just want to
       # use all conv layers for example, all Batchnorm layers,
       # or something else.


       cam_per_layer = self.compute_cam_per_layer(input_tensor)
       return self.aggregate_multi_layers(cam_per_layer)

   def __del__(self):
       self.activations_and_grads.release()

   def __enter__(self):
       return self

   def __exit__(self, exc_type, exc_value, exc_tb):
       self.activations_and_grads.release()
       if isinstance(exc_value, IndexError):
           # Handle IndexError here...
           print(
               f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
           return True

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray