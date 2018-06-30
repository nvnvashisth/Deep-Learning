"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        resnet50_32s = models.resnet50(pretrained=True)
        
        resnet_block_expansion_rate = resnet50_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()
        
        self.resnet50_32s = resnet50_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes, kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes, kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes, kernel_size=1)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x = self.resnet50_32s.maxpool(x)

        x = self.resnet50_32s.layer1(x)
        
        x = self.resnet50_32s.layer2(x)
        logits_8s = self.score_8s(x)
        
        x = self.resnet50_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet50_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample_bilinear(logits_32s, size=logits_16s_spatial_dim)
        
        logits_8s += nn.functional.upsample_bilinear(logits_16s, size=logits_8s_spatial_dim)
        
        x = nn.functional.upsample_bilinear(logits_8s, size=input_spatial_dim)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
